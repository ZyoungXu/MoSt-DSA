import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')
    rel_pos = pos[None, :] - pos[:, None]
    rel_pos += n - 1
    return rel_pos


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InterFrameLambda(nn.Module):
    def __init__(self, dim, *, dim_k, n = None, r = None, heads = 4, dim_out = None, dim_u = 1, motion_dim):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u
        self.heads = heads

        assert (dim_out % heads) == 0, "'dim_out' must be divisible by number of heads for multi-head query"
        assert (motion_dim % heads) == 0, "'motion_dim' must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)
        self.cor_embed = nn.Conv2d(2, dim_k * heads, 1, bias = True)
        self.motion_proj = nn.Conv2d(dim_k * heads, motion_dim, 1, bias = True)
        self.channel_compress = nn.Conv2d(dim_out, dim_k * heads, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x, cor):
        x = rearrange(x, 'b h w c -> b c h w')
        cor = rearrange(cor, 'b h w c -> b c h w')
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads
        x_reverse = torch.cat([x[b//2:], x[:b//2]])

        q = self.to_q(x)
        k = self.to_k(x_reverse)
        v = self.to_v(x_reverse)

        Q = self.norm_q(q)
        V = self.norm_v(v)

        Q = rearrange(Q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        V = rearrange(V, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)
        λc = einsum('b u k m, b u v m -> b k v', k, V)
        Yc = einsum('b h k n, b k v -> b h v n', Q, λc)

        if self.local_contexts:
            V = rearrange(V, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(V)
            Yp = einsum('b h k n, b k v n -> b h v n', Q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, V)
            Yp = einsum('b h k n, b n k v -> b h v n', Q, λp)

        Y = Yc + Yp
        appearance = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)

        cor_embed_ = self.cor_embed(cor)
        cor_embed = rearrange(cor_embed_, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        cor_reverse_c = einsum('b h k n, b k v -> b h v n', cor_embed, λc)
        if self.local_contexts:
            cor_reverse_p = einsum('b h k n, b k v n -> b h v n', cor_embed, λp.flatten(3))
        else:
            cor_reverse_p = einsum('b h k n, b n k v -> b h v n', cor_embed, λp)
        cor_reverse_ = rearrange(cor_reverse_c + cor_reverse_p, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        cor_reverse = self.channel_compress(cor_reverse_)
        motion = self.motion_proj(cor_reverse - cor_embed_)

        appearance = rearrange(appearance, 'b c h w -> b h w c')
        motion = rearrange(motion, 'b c h w -> b h w c')

        return appearance, motion


class StructFormerBlock(nn.Module):
    def __init__(self, dim, dim_out, dim_k, dim_u, n, r, motion_dim, heads, range, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if range == 'global':
            self.Lambda = InterFrameLambda(dim = dim,
                                        dim_k = dim_k,
                                        n = n,
                                        heads = heads,
                                        dim_out = dim_out,
                                        dim_u = dim_u,
                                        motion_dim = motion_dim)
        elif range == 'local':
            self.Lambda = InterFrameLambda(dim = dim,
                                        dim_k = dim_k,
                                        r = r,
                                        heads = heads,
                                        dim_out = dim_out,
                                        dim_u = dim_u,
                                        motion_dim = motion_dim)
        else:
            print("Range does not comply with regulations, take 'global' or 'local'!")
            sys.exit()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, cor, H, W, B):
        x_norm1 = self.norm1(x)
        x = x_norm1.view(2*B, H, W, -1)
        x_appearence, x_motion = self.Lambda(x, cor)
        x_appearence = rearrange(x_appearence, 'b h w c -> b (h w) c')
        x_motion = rearrange(x_motion, 'b h w c -> b (h w) c')
        x_appearence = x_norm1 + self.drop_path(x_appearence)
        x_appearence = x_appearence + self.drop_path(self.mlp(self.norm2(x_appearence), H, W))

        return x_appearence, x_motion


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class CrossScalePatchEmbed(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        base_dim = in_dims[0]

        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim, 1, 1)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs):
        ys = []
        k = 0
        for i in range(len(xs)):
            for _ in range(2 ** i):
                ys.append(self.layers[k](xs[-1-i]))
                k += 1

        x = self.proj(torch.cat(ys,1))

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.norm(x)

        return x, H, W


class StructFormer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16],
                 mlp_ratios=[4, 4], lambda_global_or_local = 'local', lambda_dim_k = 16, lambda_dim_u = 1,
                 lambda_n = 32, lambda_r = None, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 4, 4], **kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = CrossScalePatchEmbed(embed_dims[:i],
                                                        embed_dim=embed_dims[i])
                        block = nn.ModuleList([StructFormerBlock(
                            dim=embed_dims[i], dim_out=embed_dims[i], dim_k=lambda_dim_k, dim_u=lambda_dim_u, n=lambda_n, r=lambda_r,
                            motion_dim=motion_dims[i], heads=num_heads[i-self.conv_stages], range = lambda_global_or_local, mlp_ratio=mlp_ratios[i-self.conv_stages],
                            drop=drop_rate, drop_path=dpr[cur + j])
                            for j in range(depths[i])])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                        block = nn.ModuleList([StructFormerBlock(
                            dim=embed_dims[i], dim_out=embed_dims[i], dim_k=lambda_dim_k, dim_u=lambda_dim_u, n=int(lambda_n/2), r=lambda_r,
                            motion_dim=motion_dims[i], heads=num_heads[i-self.conv_stages], range = lambda_global_or_local, mlp_ratio=mlp_ratios[i-self.conv_stages],
                            drop=drop_rate, drop_path=dpr[cur + j])
                            for j in range(depths[i])])

                    norm = norm_layer(embed_dims[i])
                    setattr(self, f"norm{i + 1}", norm)

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0]

        x = torch.cat([x1, x2], 0)

        motion_features = []
        appearence_features = []
        xs = []

        for i in range(self.num_stages):
            motion_features.append([])
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            norm = getattr(self, f"norm{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
                xs.append(x)
            else:
                if i == self.conv_stages:
                    x, H, W = patch_embed(xs)
                else:
                    x, H, W = patch_embed(x)

                cor = self.get_cor((x.shape[0], H, W), x.device)

                for blk in block:
                    x, x_motion = blk(x, cor, H, W, B)

                    motion_features[i].append(x_motion.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous())

                x = norm(x)
                x = x.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                motion_features[i] = torch.cat(motion_features[i], 1)

            appearence_features.append(x)
        return appearence_features, motion_features


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x


def encoder(**kargs):
    model = StructFormer(**kargs)
    return model
