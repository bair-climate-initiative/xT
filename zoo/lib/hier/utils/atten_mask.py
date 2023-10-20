import numpy as np
import torch

# import xformers.ops as xops
# from xformers.components.attention import ScaledDotProduct


def build_attention_mask_adjacent(input_resolution, radius, grid=None):
    """
    input_shape: tuple
    """
    abs_coords = [
        torch.arange(x, dtype=torch.float32) for x in input_resolution
    ]
    abs_coords = torch.stack(
        torch.meshgrid(abs_coords), dim=-1
    ).contiguous()  # T L H W 4
    total_len = np.product(input_resolution)
    n_dim = len(input_resolution)
    abs_coords = abs_coords.view(total_len, n_dim)  # (T L H W) 4
    abs_coords_diff = torch.abs(abs_coords[:, None] - abs_coords[None, :]).sum(
        -1
    )  # N N 4
    mask = abs_coords_diff <= radius

    do_grid = grid is not None and len(grid) > 0
    if do_grid:
        grid_indices = [(torch.arange(x) + 0.5) / x for x in grid]
        grid_indices = [x * y for x, y in zip(grid_indices, input_resolution)]
        grid_indices = [x.int() for x in grid_indices]
        grid_indices = torch.stack(
            torch.meshgrid(grid_indices), dim=-1
        ).contiguous()
        grid_indices = grid_indices.view(-1, n_dim).long()
        if n_dim == 4:
            i, j, k, w = grid_indices.T
            base_mask = torch.zeros(*input_resolution, dtype=bool)
            base_mask[i, j, k, w] = True
            base_mask = base_mask.view(-1)
            in_grid = base_mask[None,] & base_mask[:, None]
            mask = mask | in_grid
        else:
            raise NotImplemented
    mask = torch.where(
        mask,
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(float("-inf"), dtype=torch.float32),
    )  # 1 if in, 0 if out
    return mask


if __name__ == "__main__":
    input_resolution = (4, 8, 32, 64)
    B, M, K = 3, np.product(input_resolution), 128
    kwargs = dict(device="cuda", dtype=torch.float16)
    attention = ScaledDotProduct().cuda()
    q = torch.randn([B, M, 8, K], **kwargs)
    k = torch.randn([B, M, 8, K], **kwargs)
    v = torch.randn([B, M, 8, K], **kwargs)
    mask0 = build_attention_mask_adjacent(input_resolution, 1).to("cuda").half()
    # mask = mask0[None,].expand(B*8,-1,-1)
    # out_gqa = xops.memory_efficient_attention(
    # q,
    # k,
    # v,
    # attn_bias=mask
    # )
    # torch.cuda.synchronize()
    # max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    # print(f"Dense - Peak memory use: {max_memory}MB")
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    out_gqa = attention(q=q, k=k, v=v, mask=mask0)
    max_memory = torch.cuda.max_memory_allocated() // 2**20
    print(f"Sparse - Peak memory use : {max_memory}MB")
