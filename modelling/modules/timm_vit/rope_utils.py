
import torch 

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis

def compute_axial_cis_1d(dim: int, end: int, theta: float = 100.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = init_t(end)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def compute_axial_cis_2d(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def compute_axial_cis_3d(
    dim: int, end_x: int, end_y: int, end_t: int,
    theta_xy: float = 100.0, theta_t: float = 100.0,
    stride_xy: int = 1, stride_t: int = 1, fps: float = None,
    frame_pts: list = None
):
    ch_xy = dim // 6
    ch_t = dim // 2 - ch_xy * 2
    freqs_xy = 1.0 / (theta_xy ** (torch.arange(0, ch_xy).float() * 6 / dim))
    freqs_t = 1.0 / (theta_t ** (torch.arange(0, ch_t).float() * 6 / dim))

    assert end_x % stride_xy == 0 and end_y % stride_xy == 0 and end_t % stride_t == 0
    new_x, new_y, new_t = int(end_x / stride_xy), int(end_y / stride_xy), int(end_t / stride_t)
    offset_xy, offset_t = (stride_xy - 1) / 2, (stride_t - 1) / 2

    # t_x, t_y, t_t = init_t_xyt(end_x, end_y, end_t)

    pos_x = torch.arange(new_x).float() * stride_xy + offset_xy
    pos_y = torch.arange(new_y).float() * stride_xy + offset_xy
    pos_t = torch.arange(new_t).float() * stride_t + offset_t
    if frame_pts is not None:
        assert stride_t == 1
        pos_t = pos_t[frame_pts]
    tg, yg, xg = torch.meshgrid(pos_t, pos_y, pos_x, indexing='ij')
    t_x = xg.reshape(-1)
    t_y = yg.reshape(-1)
    t_t = tg.reshape(-1)

    if fps is not None:
        t_t = t_t * 24 / fps # here we use fps=24 as the anchor value

    freqs_x = torch.outer(t_x, freqs_xy)
    freqs_y = torch.outer(t_y, freqs_xy)
    freqs_t = torch.outer(t_t, freqs_t)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    freqs_cis_t = torch.polar(torch.ones_like(freqs_t), freqs_t)

    return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_t], dim=-1)

def init_t(end: int):
    t = torch.arange(end, dtype=torch.float32)
    return t

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def init_t_xyt(end_x: int, end_y: int, end_t: int):
    t = torch.arange(end_x * end_y * end_t, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = (torch.div(t, end_x, rounding_mode='floor') % end_y).float()
    t_t = torch.div(t, end_x * end_y, rounding_mode='floor')
    return t_x, t_y, t_t

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise NotImplementedError()
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def apply_rotary_emb_partial(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    只对前面 freqs_cis.shape[-1] 个 complex-channel 应用 RoPE，剩余通道保持不动。
    xq, xk: [B, S, n_heads, head_dim]
    freqs_cis: broadcast 到 xq_ 之后，shape [B, S, n_heads, M] (complex)
    """
    # 1) 先把 real tensor 重塑成 complex 张量，最后一维是 complex 维度
    B, S, H, D = xq.shape
    Cc = D // 2  # complex-channel 数
    xq_ = torch.view_as_complex(xq.float().reshape(B, S, H, Cc, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(B, S, H, Cc, 2))

    # 2) 取出要旋转的 complex-channel 数
    M = freqs_cis.shape[-1]
    assert M <= Cc, f"freqs_cis={M} should be lesser than complex-channel={Cc}"

    # 3) 广播 freqs_cis 到 [B, S, H, M]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_[..., :M])  # shape (B,S,H,M)

    # 4) 拆分：前 M 个做旋转，剩余 Cc-M 个保持不动
    q_rot_part = xq_[..., :M] * freqs_cis
    k_rot_part = xk_[..., :M] * freqs_cis
    q_pass_part = xq_[..., M:]
    k_pass_part = xk_[..., M:]

    # 5) 拼回 complex 张量
    xq_out_complex = torch.cat([q_rot_part, q_pass_part], dim=-1)
    xk_out_complex = torch.cat([k_rot_part, k_pass_part], dim=-1)

    # 6) 转回 real，并恢复原始 shape
    xq_out = torch.view_as_real(xq_out_complex) \
                   .reshape(B, S, H, D) \
                   .type_as(xq) \
                   .to(xq.device)
    xk_out = torch.view_as_real(xk_out_complex) \
                   .reshape(B, S, H, D) \
                   .type_as(xk) \
                   .to(xk.device)

    return xq_out, xk_out