import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import process_group_manager as pgm

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply usage of rotary embeddings to q and k.
    Expects q, k in shape [batch, heads, seq, dim]
    Expects cos, sin in shape [seq, dim]
    """
    # Reshape cos/sin for broadcasting: [seq, dim] -> [1, 1, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Standard RMSNorm implementation
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim % 2 == 0
    # Frequency calculation on CPU
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    position = torch.arange(seq_length).to(device).unsqueeze(1).float() # [seq_length, 1]
    theta = theta.to(device)
    
    # Returns [seq_length, head_dim]
    cos = torch.cos(position.float() * theta.float()).to(dtype).repeat(1, 2)
    sin = torch.sin(position.float() * theta.float()).to(dtype).repeat(1, 2)
    return cos, sin

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert config.num_attention_heads % pgm.process_group_manager.tp_world_size == 0, "num_attention_heads should be divisible by tp world size"
        assert config.num_key_value_heads % pgm.process_group_manager.tp_world_size == 0, "num_key_value_heads should be divisible by  tp world size"
        self.num_local_heads = config.num_attention_heads // pgm.process_group_manager.tp_world_size # TP parallelism
        self.num_local_kv_heads = config.num_key_value_heads // pgm.process_group_manager.tp_world_size # TP parallelism
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer_idx = layer_idx
        
    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_dim = x.size()
        
        q = self.q_proj(x) 
        k = self.k_proj(x) 
        v = self.v_proj(x) 

        # Reshape to [batch, seq, heads, dim] -> Transpose to [batch, heads, seq, dim]
        q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Embeddings (Manual implementation)
        # We slice cos/sin to match head_dim because get_cos_sin might produce larger caches in some implementations
        # (though in your code it matches exactly).
        q, k = apply_rotary_pos_emb(q, k, cos[:, :self.head_dim], sin[:, :self.head_dim])
        
        # Repeat KV heads if using Grouped Query Attention (GQA)
        k = k.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) 
        v = v.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) 
        
        # --- 3. Replacement for Flash Attention Func ---
        # If query and key lengths match, it implies training or full-sequence processing -> usually causal
        is_causal = True if q.size(2) == k.size(2) else False 
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, # SDPA handles causal masking via the is_causal flag
            dropout_p=0.0,
            is_causal=is_causal
        )
        
        # Reshape back: [batch, heads, seq, dim] -> [batch, seq, heads, dim] -> [batch, seq, hidden]
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.num_local_heads * self.head_dim)
        out = self.out_proj(out) 
        return out

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    # TritonRMSNorm (now standard RMSNorm) -> Attention -> Residual -> TritonRMSNorm -> MLP -> Residual
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config, layer_idx = layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx
        head_dim = config.hidden_size // config.num_attention_heads
        # [max_position_embeddings, head_dim]
        self.cos, self.sin = get_cos_sin(config.max_position_embeddings, head_dim=head_dim , base=config.rope_theta) 

    def forward(self, x, attention_mask = None, position_ids = None):
        cos, sin = self.cos, self.sin 
        x = x + self.attention(self.input_layernorm(x), cos, sin, attention_mask, position_ids) # Attention 
        x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
        return x
    
class Llama(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check 
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0 
        
        # params
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads 
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config
        
        # modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)])
        self.final_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.final_norm = TritonRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)  # [batch_size, seq_length, hidden_dim]
        x = self.final_norm(x)
        logits = self.final_proj(x)
        
        return logits  # [batch_size, seq_length, vocab_size]