import torch
import numpy as np


def occupancy_activation(alpha, distances=None): #体密度的激活函数，给体密度加上一个sigmoid函数
    # occ = 1.0 - torch.exp(-alpha * distances)
    occ = torch.sigmoid(alpha)    # unisurf

    return occ

def alpha_to_occupancy(depths, dirs, alpha, add_last=False):
    interval_distances = depths[..., 1:] - depths[..., :-1] #计算每个点的距离
    if add_last:
        last_distance = torch.empty(
            (depths.shape[0], 1),
            device=depths.device,
            dtype=depths.dtype).fill_(0.1)
        interval_distances = torch.cat(
            [interval_distances, last_distance], dim=-1)

    dirs_norm = torch.norm(dirs, dim=-1)
    interval_distances = interval_distances * dirs_norm[:, None]  #这个好像没有用到
    occ = occupancy_activation(alpha, interval_distances) #给alpha加上一个sigmoid函数，就变成了occ体密度

    return occ

def occupancy_to_termination(occupancy, is_batch=False): #计算终止概率
    if is_batch:
        first = torch.ones(list(occupancy.shape[:2]) + [1], device=occupancy.device)
        free_probs = (1. - occupancy + 1e-10)[:, :, :-1]
    else:
        first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
        free_probs = (1. - occupancy + 1e-10)[:, :-1]
    free_probs = torch.cat([first, free_probs], dim=-1)
    term_probs = occupancy * torch.cumprod(free_probs, dim=-1) #计算终止概率,torch.cumprod是累乘,就是论文里的公式Ti = o ( xi )∏j < i ( 1-o ( xj) )。

    # using escape probability
    # occupancy = occupancy[:, :-1]
    # first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    # free_probs = (1. - occupancy + 1e-10)
    # free_probs = torch.cat([first, free_probs], dim=-1)
    # last = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    # occupancy = torch.cat([occupancy, last], dim=-1)
    # term_probs = occupancy * torch.cumprod(free_probs, dim=-1)

    return term_probs

def render(termination, vals, dim=-1):
    weighted_vals = termination * vals
    render = weighted_vals.sum(dim=dim) #论文里的公式（3），加上权重

    return render #返回渲染结果

def render_loss(render, gt, loss="L1", normalise=False):
    residual = render - gt
    if loss == "L2":
        loss_mat = residual ** 2
    elif loss == "L1":
        loss_mat = torch.abs(residual)
    else:
        print("loss type {} not implemented!".format(loss))

    if normalise:
        loss_mat = loss_mat / gt

    return loss_mat #返回loss

def reduce_batch_loss(loss_mat, var=None, avg=True, mask=None, loss_type="L1"):
    mask_num = torch.sum(mask, dim=-1)
    if (mask_num == 0).any():   # no valid sample, return 0 loss，如果mask_num中有0，说明没有有效的样本，返回0
        loss = torch.zeros_like(loss_mat)
        if avg:
            loss = torch.mean(loss, dim=-1)
        return loss
    if var is not None:
        eps = 1e-4
        if loss_type == "L2":
            information = 1.0 / (var + eps)
        elif loss_type == "L1":
            information = 1.0 / (torch.sqrt(var) + eps)

        loss_weighted = loss_mat * information #如果有方差，给loss加上方差，应该是受NICE-SLAM的loss的启发
    else:
        loss_weighted = loss_mat

    if avg: #如果avg为True，则计算平均loss
        if mask is not None:
            loss = (torch.sum(loss_weighted, dim=-1)/(torch.sum(mask, dim=-1)+1e-10)) #如果有mask，则除以mask计算平均loss
            if (loss > 100000).any():
                print("loss explode")
                exit(-1)
        else:
            loss = torch.mean(loss_weighted, dim=-1).sum()
    else:
        loss = loss_weighted

    return loss

def make_3D_grid(occ_range=[-1., 1.], dim=256, device="cuda:0", transform=None, scale=None): #生成3D网格点
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim, device=device)
    grid = torch.meshgrid(t, t, t) #生成网格点
    grid_3d = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d
