import torch
import model
import embedding
import render_rays
import numpy as np
import vis
from tqdm import tqdm

class Trainer:
    def __init__(self, cfg):
        self.obj_id = cfg.obj_id #物体的id
        self.device = cfg.training_device # cuda
        self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 256 for iMAP, 128 for seperate bg
        self.obj_scale = cfg.obj_scale # 10 for bg and iMAP
        self.n_unidir_funcs = cfg.n_unidir_funcs  #5，暂时不知道是什么
        self.emb_size1 = 21*(3+1)+3 #embedding size for obj
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1

        self.load_network() #加载网络

        if self.obj_id == 0:
            self.bound_extent = 0.995  #边界范围
        else:
            self.bound_extent = 0.9

    def load_network(self): #加载网络
        self.fc_occ_map = model.OccupancyMap(
            self.emb_size1,
            self.emb_size2,
            hidden_size=self.hidden_feature_size
        ) #网络结构
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def meshing(self, bound, obj_center, grid_dim=256): #建立网格mesh
        occ_range = [-1., 1.] #体密度是-1到1
        range_dist = occ_range[1] - occ_range[0] #2
        scene_scale_np = bound.extent / (range_dist * self.bound_extent) #bound.extent是边界的范围，self.bound_extent是边界范围的比例
        scene_scale = torch.from_numpy(scene_scale_np).float().to(self.device)
        transform_np = np.eye(4, dtype=np.float32)
        transform_np[:3, 3] = bound.center #边界框的中心
        transform_np[:3, :3] = bound.R #边界的旋转
        # transform_np = np.linalg.inv(transform_np)  #
        transform = torch.from_numpy(transform_np).to(self.device)
        grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                           scale=scene_scale, transform=transform).view(-1, 3) #生成网格点
        grid_pc -= obj_center.to(grid_pc.device) #减去物体的中心，这样可以使得物体的中心在原点
        ret = self.eval_points(grid_pc) #计算网格点的体密度和颜色
        if ret is None:
            return None

        occ, _ = ret #获取体密度
        mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().numpy()) #生成网格
        if mesh is None:
            print("marching cube failed")
            return None

        # Transform to [-1, 1] range，将网格点的范围变为-1到1
        mesh.apply_translation([-0.5, -0.5, -0.5])
        mesh.apply_scale(2)

        # Transform to scene coordinates，将网格点的范围变为边界的范围
        mesh.apply_scale(scene_scale_np)
        mesh.apply_transform(transform_np)

        vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
        ret = self.eval_points(vertices_pts) #计算网格点的体密度和颜色
        if ret is None:
            return None
        _, color = ret # 获取颜色
        mesh_color = color * 255
        vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        return mesh

    def eval_points(self, points, chunk_size=100000): #输入点云，渲染体密度和颜色
        # 256^3 = 16777216
        alpha, color = [], []
        n_chunks = int(np.ceil(points.shape[0] / chunk_size)) #计算chunk的数量
        with torch.no_grad():
            for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size) #每次取出100000个点
                embedding_k = self.pe(points[chunk_idx, ...]) #计算每个点的embedding
                alpha_k, color_k = self.fc_occ_map(embedding_k) #输入embedding计算每个点的体密度和颜色
                alpha.extend(alpha_k.detach().squeeze())
                color.extend(color_k.detach().squeeze())
        alpha = torch.stack(alpha)
        color = torch.stack(color)

        occ = render_rays.occupancy_activation(alpha).detach()
        if occ.max() == 0:
            print("no occ")
            return None
        return (occ, color) #返回体密度和颜色











