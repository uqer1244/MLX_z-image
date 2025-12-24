import mlx.core as mx
import mlx.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dims)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None: mid_size = out_size
        self.linear1 = nn.Linear(frequency_embedding_size, mid_size)
        self.linear2 = nn.Linear(mid_size, out_size)
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        t = t.astype(mx.float32)
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(0, half, dtype=mx.float32) / half)
        args = (t[:, None] * freqs[None, :])
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=1)
        return self.linear2(nn.silu(self.linear1(embedding)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        # FFN has little room for optimization (already a dense MatMul block)
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim: int, nheads: int, rope_theta: float = 256.0, eps: float = 1e-5):
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        # Pre-compute frequencies for each split section
        # Section 1: 32 dim (0~15 freq)
        # Section 2: 48 dim (0~23 freq)
        # Section 3: 48 dim (0~23 freq)
        self.dims = [32, 48, 48]
        self.splits = [0, 32, 80]  # Start indices
        self.freqs_cache = {}

    def _get_fused_args(self, positions):
        """
        [Optimization Core]
        Instead of splitting Q/K data, we pre-calculate and fuse the 'angles (Args)'.
        Manipulating angles (KB size) is much faster than manipulating data tensors (MB size).
        """
        # positions: (1, L, 3) -> H, W, T
        B, L, _ = positions.shape

        # Cache Key: Reuse if sequence length L hasn't changed
        if L in self.freqs_cache:
            freqs_tuple = self.freqs_cache[L]
        else:
            # Pre-compute frequencies (Executed once)
            freqs_list = []
            for d in self.dims:
                half = d // 2
                f = mx.exp(-mx.log(256.0) * mx.arange(0, half, dtype=mx.float32) / half)
                freqs_list.append(f)  # [ (16,), (24,), (24,) ]
            self.freqs_cache[L] = freqs_list
            freqs_tuple = freqs_list

        # Calculate angles (Theta) for each section
        # pos: (1, L)
        # freqs: (D_half,)
        # args: (1, L, 1, D_half)

        # 1. Height Section (Dims 0~32)
        pos_h = positions[..., 0].astype(mx.float32)
        args_h = pos_h[..., None, None] * freqs_tuple[0][None, None, None, :]

        # 2. Width Section (Dims 32~80)
        pos_w = positions[..., 1].astype(mx.float32)
        args_w = pos_w[..., None, None] * freqs_tuple[1][None, None, None, :]

        # 3. Time Section (Dims 80~128)
        pos_t = positions[..., 2].astype(mx.float32)
        args_t = pos_t[..., None, None] * freqs_tuple[2][None, None, None, :]

        # Fuse 'angles' here. (Concatenate Args, NOT Tensors)
        # Result: (1, L, 1, 64) -> Half of the total head dimension
        return mx.concatenate([args_h, args_w, args_t], axis=-1)

    def __call__(self, x, mask=None, positions=None, cos=None, sin=None):
        B, L, D = x.shape

        q = self.to_q(x).reshape(B, L, self.nheads, self.head_dim)
        k = self.to_k(x).reshape(B, L, self.nheads, self.head_dim)
        v = self.to_v(x).reshape(B, L, self.nheads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if cos is None or sin is None:
            if positions is not None:
                # 1. Get fused angles (Args)
                args = self._get_fused_args(positions)  # (1, L, 1, D/2)

                # 2. Calculate Sin/Cos (Performed once for the entire chunk)
                cos = mx.cos(args)
                sin = mx.sin(args)

        if cos is not None and sin is not None:
            # 3. Rotate entire tensor (No Split, No Loop)
            # [cite_start]Indexing 0::2 and 1::2 are memory views[cite: 1], so copy cost is near zero.
            q1 = q[..., 0::2]
            q2 = q[..., 1::2]
            q = mx.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1).reshape(B, L, self.nheads, self.head_dim)

            k1 = k[..., 0::2]
            k2 = k[..., 1::2]
            k = mx.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1).reshape(B, L, self.nheads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.to_out(output.transpose(0, 2, 1, 3).reshape(B, L, D))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, config, layer_id, modulation=True):
        super().__init__()
        dim = config['dim']
        nheads = config['nheads']
        self.modulation = modulation
        self.attention = Attention(dim, nheads, rope_theta=config.get('rope_theta', 256.0), eps=1e-5)
        self.feed_forward = FeedForward(dim, int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim)
        self.ffn_norm1 = RMSNorm(dim)
        self.attention_norm2 = RMSNorm(dim)
        self.ffn_norm2 = RMSNorm(dim)
        if modulation: self.adaLN_modulation = nn.Linear(256, 4 * dim, bias=True)

    def __call__(self, x, mask, positions, adaln_input=None, cos=None, sin=None):
        if self.modulation:
            chunks = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(chunks, 4, axis=-1)
            scale_msa, gate_msa = scale_msa[..., None, :], gate_msa[..., None, :]
            scale_mlp, gate_mlp = scale_mlp[..., None, :], gate_mlp[..., None, :]

            norm_x = self.attention_norm1(x) * (1 + scale_msa)
            x = x + mx.tanh(gate_msa) * self.attention_norm2(self.attention(norm_x, mask, positions, cos, sin))

            norm_ffn = self.ffn_norm1(x) * (1 + scale_mlp)
            x = x + mx.tanh(gate_mlp) * self.ffn_norm2(self.feed_forward(norm_ffn))
        else:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), mask, positions, cos, sin))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(256, dim, bias=True))

    def __call__(self, x, c):
        scale = self.adaLN_modulation.layers[1](self.adaLN_modulation.layers[0](c))
        return self.linear(self.norm_final(x) * (1 + scale[:, None, :]))


class ZImageTransformerMLX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config['dim']
        self.t_scale = config.get('t_scale', 1000.0)
        self.t_embedder = TimestepEmbedder(256, mid_size=1024)
        self.x_embedder = nn.Linear(config['in_channels'] * 4, dim, bias=True)
        self.cap_embedder = nn.Sequential(RMSNorm(config['cap_feat_dim']),
                                          nn.Linear(config['cap_feat_dim'], dim, bias=True))
        self.final_layer = FinalLayer(dim, config['in_channels'] * 4)

        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))

        self.noise_refiner = [ZImageTransformerBlock(config, i, True) for i in range(config['n_refiner_layers'])]
        self.context_refiner = [ZImageTransformerBlock(config, i, False) for i in range(config['n_refiner_layers'])]
        self.layers = [ZImageTransformerBlock(config, i, True) for i in range(config['n_layers'])]

    def prepare_rope(self, positions):
        args = self.layers[0].attention._get_fused_args(positions)
        return mx.cos(args), mx.sin(args)

    def __call__(self, x, t, cap_feats, x_pos, cap_pos, cos=None, sin=None, x_mask=None, cap_mask=None):
        temb = self.t_embedder(t * self.t_scale)
        x = self.x_embedder(x)
        if x_mask is not None: x = mx.where(x_mask[..., None], self.x_pad_token, x)

        cap_feats = self.cap_embedder.layers[1](self.cap_embedder.layers[0](cap_feats))
        if cap_mask is not None: cap_feats = mx.where(cap_mask[..., None], self.cap_pad_token, cap_feats)

        x_len = x.shape[1]
        x_cos, x_sin, cap_cos, cap_sin = None, None, None, None
        if cos is not None and sin is not None:
             x_cos = cos[:, :x_len, ...]
             x_sin = sin[:, :x_len, ...]
             cap_cos = cos[:, x_len:, ...]
             cap_sin = sin[:, x_len:, ...]

        for l in self.noise_refiner: x = l(x, None, x_pos, temb, cos=x_cos, sin=x_sin)
        for l in self.context_refiner: cap_feats = l(cap_feats, None, cap_pos, None, cos=cap_cos, sin=cap_sin)

        unified = mx.concatenate([x, cap_feats], axis=1)
        unified_pos = mx.concatenate([x_pos, cap_pos], axis=1)

        unified_mask = None
        for l in self.layers: unified = l(unified, unified_mask, unified_pos, temb, cos=cos, sin=sin)
        return self.final_layer(unified[:, :x.shape[1], :], temb)