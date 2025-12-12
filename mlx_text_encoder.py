import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dims)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config["num_key_value_heads"]
        self.rope_theta = config.get("rope_theta", 1000000.0)

        # Projection (Bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q/K Norm (Head Dim 기준)
        self.q_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))

    def __call__(self, x, mask=None):
        B, L, D = x.shape

        # 1. Projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for Norm (B, L, H, D)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim)

        # 3. Apply Q/K Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. RoPE
        def apply_rope(x, theta):
            dim = x.shape[-1]
            half = dim // 2
            freqs = mx.exp(-mx.log(theta) * mx.arange(0, half, dtype=mx.float32) / half)
            pos = mx.arange(L, dtype=mx.float32)

            args = pos[:, None] * freqs[None, :]
            cos = mx.cos(args).reshape(L, 1, half)
            sin = mx.sin(args).reshape(L, 1, half)

            x1 = x[..., :half]
            x2 = x[..., half:]
            return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        q = apply_rope(q, self.rope_theta)
        k = apply_rope(k, self.rope_theta)

        # 5. Transpose for Attention (B, H, L, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # [핵심 수정] GQA (Grouped Query Attention) 처리
        # Key/Value 헤드 수가 Query 헤드 수보다 적을 경우, 복제(Repeat)하여 맞춰줍니다.
        n_rep = self.num_heads // self.num_key_value_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # 6. Attention Calculation
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            scores = scores + mask

        probs = mx.softmax(scores, axis=-1)
        output = (probs @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, x, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [TransformerBlock(config) for _ in range(config["num_hidden_layers"])]
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, input_ids):
        x = self.embed_tokens(input_ids)
        B, L = input_ids.shape
        mask = mx.triu(mx.full((L, L), -1e9), k=1)

        hidden_states = []
        for layer in self.layers:
            x = layer(x, mask)
            hidden_states.append(x)

        final_out = self.norm(x)
        hidden_states.append(final_out)

        return hidden_states[-2]


class TextEncoderMLX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)

    def __call__(self, input_ids):
        return self.model(input_ids)