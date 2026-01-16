import numpy as np

from scipy.ndimage import zoom

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    # GELU 近似实现
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

class Attention:
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # 权重占位符
        self.qkv_weight = None
        self.qkv_bias = None
        self.proj_weight = None
        self.proj_bias = None

    def forward(self, x):
        B, N, C = x.shape
        # QKV 计算
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        
        # Reshape: (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = softmax(attn, axis=-1)

        # 组合 heads
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        
        # 输出投影
        x = x @ self.proj_weight.T + self.proj_bias
        return x

class MLP:
    def __init__(self, in_features, hidden_features):
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None

    def forward(self, x):
        x = x @ self.fc1_weight.T + self.fc1_bias
        x = gelu(x)
        x = x @ self.fc2_weight.T + self.fc2_bias
        return x

class Block:
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        self.norm1_gamma = None
        self.norm1_beta = None
        self.attn = Attention(dim, num_heads)
        
        self.norm2_gamma = None
        self.norm2_beta = None
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.layer_scale1 = None
        self.layer_scale2 = None

    def forward(self, x):
        # Result connection + Pre-Norm
        attn_out = self.attn.forward(layer_norm(x, self.norm1_gamma, self.norm1_beta))
        if self.layer_scale1 is not None:
            attn_out = attn_out * self.layer_scale1
        x = x + attn_out

        mlp_out = self.mlp.forward(layer_norm(x, self.norm2_gamma, self.norm2_beta))
        if self.layer_scale2 is not None:
            mlp_out = mlp_out * self.layer_scale2
        x = x + mlp_out
        return x

class PatchEmbed:
    def __init__(self, patch_size=14, in_chans=3, embed_dim=384):
        self.patch_size = patch_size
        self.proj_weight = None # (embed_dim, in_chans, patch_size, patch_size)
        self.proj_bias = None

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        P = self.patch_size
        if H % P != 0 or W % P != 0:
            raise ValueError(f"Input size ({H}, {W}) must be divisible by patch size {P}.")
        
        # 这里的实现模拟 Conv2d(stride=patch_size)
        # 将图像重塑为 patches
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.transpose(0, 2, 4, 1, 3, 5) # (B, H/P, W/P, C, P, P)
        x = x.reshape(B, -1, C * P * P)   # (B, N, C*P*P)
        
        # 线性投影
        W_flat = self.proj_weight.reshape(self.proj_weight.shape[0], -1)
        x = x @ W_flat.T + self.proj_bias
        return x

class DinoV2:
    def __init__(self, layers=12, embed_dim=384, num_heads=6, patch_size=14):
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = None
        self.pos_embed = None
        self.blocks = [Block(embed_dim, num_heads) for _ in range(layers)]
        self.norm_gamma = None
        self.norm_beta = None

    def interpolate_pos_encoding(self, x, H, W):
        # 位置编码插值逻辑，支持不同分辨率的输入
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        w0 = W // self.patch_embed.patch_size
        h0 = H // self.patch_embed.patch_size
        
        # 简单的双线性插值实现
        N_sqrt = int(np.sqrt(N))
        data = patch_pos_embed.reshape(1, N_sqrt, N_sqrt, dim)
        scale_h = h0 / N_sqrt
        scale_w = w0 / N_sqrt
        new_pos = zoom(data, (1, scale_h, scale_w, 1), order=3)
        new_pos = new_pos.reshape(1, -1, dim)
        return np.concatenate((class_pos_embed[:, None, :], new_pos), axis=1)

    def load_from_npz(self, weights_source):
        if isinstance(weights_source, str):
            with np.load(weights_source) as data:
                weights = {k: data[k] for k in data.files}
        elif isinstance(weights_source, np.lib.npyio.NpzFile):
            weights = {k: weights_source[k] for k in weights_source.files}
        else:
            weights = weights_source

        self.patch_embed.proj_weight = weights['embeddings.patch_embeddings.projection.weight']
        self.patch_embed.proj_bias = weights['embeddings.patch_embeddings.projection.bias']
        self.cls_token = weights['embeddings.cls_token']
        self.pos_embed = weights['embeddings.position_embeddings']
        self.norm_gamma = weights['layernorm.weight']
        self.norm_beta = weights['layernorm.bias']

        for idx, block in enumerate(self.blocks):
            prefix = f'encoder.layer.{idx}.'
            block.norm1_gamma = weights[prefix + 'norm1.weight']
            block.norm1_beta = weights[prefix + 'norm1.bias']
            q_w = weights[prefix + 'attention.attention.query.weight']
            k_w = weights[prefix + 'attention.attention.key.weight']
            v_w = weights[prefix + 'attention.attention.value.weight']
            block.attn.qkv_weight = np.concatenate([q_w, k_w, v_w], axis=0)
            q_b = weights[prefix + 'attention.attention.query.bias']
            k_b = weights[prefix + 'attention.attention.key.bias']
            v_b = weights[prefix + 'attention.attention.value.bias']
            block.attn.qkv_bias = np.concatenate([q_b, k_b, v_b], axis=0)
            block.attn.proj_weight = weights[prefix + 'attention.output.dense.weight']
            block.attn.proj_bias = weights[prefix + 'attention.output.dense.bias']
            block.norm2_gamma = weights[prefix + 'norm2.weight']
            block.norm2_beta = weights[prefix + 'norm2.bias']
            block.mlp.fc1_weight = weights[prefix + 'mlp.fc1.weight']
            block.mlp.fc1_bias = weights[prefix + 'mlp.fc1.bias']
            block.mlp.fc2_weight = weights[prefix + 'mlp.fc2.weight']
            block.mlp.fc2_bias = weights[prefix + 'mlp.fc2.bias']
            block.layer_scale1 = weights.get(prefix + 'layer_scale1.lambda1')
            block.layer_scale2 = weights.get(prefix + 'layer_scale2.lambda1')

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.patch_embed.forward(x)
        
        # 拼接 CLS token
        cls_token = np.tile(self.cls_token, (B, 1, 1))
        x = np.concatenate((cls_token, x), axis=1)
        
        # 添加位置编码
        x = x + self.interpolate_pos_encoding(x, H, W)
        
        # Blocks forward
        for blk in self.blocks:
            x = blk.forward(x)
            
        # 最后的 Layer Norm
        x = layer_norm(x, self.norm_gamma, self.norm_beta)
        
        # 返回 CLS token 的特征
        return x[:, 0]

class Dinov2Numpy:
    def __init__(self, weights_source):
        if isinstance(weights_source, str):
            with np.load(weights_source) as data:
                weights = {k: data[k] for k in data.files}
        elif isinstance(weights_source, np.lib.npyio.NpzFile):
            weights = {k: weights_source[k] for k in weights_source.files}
        else:
            weights = weights_source

        embed_dim = weights['embeddings.cls_token'].shape[-1]
        patch_size = weights['embeddings.patch_embeddings.projection.weight'].shape[-1]
        depth = len([k for k in weights.keys() if k.endswith('norm1.weight') and k.startswith('encoder.layer.')])
        num_heads = embed_dim // 64

        self.model = DinoV2(layers=depth, embed_dim=embed_dim, num_heads=num_heads, patch_size=patch_size)
        self.model.load_from_npz(weights)

    def __call__(self, pixel_values):
        return self.model.forward(pixel_values)


# Example usage
if __name__ == "__main__":
    vit = Dinov2Numpy('vit-dinov2-base.npz')
    dummy_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
    features = vit(dummy_img)
    print("Dummy forward pass finished:", features.shape)
