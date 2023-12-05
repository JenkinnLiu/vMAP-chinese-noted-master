import random
import numpy as np
import torch
from time import perf_counter_ns
from tqdm import tqdm
import trainer
import open3d
import trimesh
import scipy
from bidict import bidict
import copy
import os

import utils


class performance_measure:

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, type, value, tb):
        self.end_time = perf_counter_ns()
        self.exec_time = self.end_time - self.start_time

        print(f"{self.name} excution time: {(self.exec_time)/1000000:.2f} ms")

def origin_dirs_W(T_WC, dirs_C):

    assert T_WC.shape[0] == dirs_C.shape[0]
    assert T_WC.shape[1:] == (4, 4)
    assert dirs_C.shape[2] == 3 # dir_C是相机坐标系下的dirs

    dirs_W = (T_WC[:, None, :3, :3] @ dirs_C[..., None]).squeeze() #获取世界坐标系下的射线的方向dirs_W，将dirs_C转换到世界坐标系下

    origins = T_WC[:, :3, -1] #射线的起点是相机的位置

    return origins, dirs_W #返回世界坐标系下的dirs_W和origins


# @torch.jit.script
def stratified_bins(min_depth, max_depth, n_bins, n_rays, type=torch.float32, device = "cuda:0"):
    # type: (Tensor, Tensor, int, int) -> Tensor

    bin_limits_scale = torch.linspace(0, 1, n_bins+1, dtype=type, device=device) #bin是用于存储深度的，这里是将深度分成n_bins个bin，每个bin的深度范围是[0,1]、
    #论文中的均匀采样

    if not torch.is_tensor(min_depth):
        min_depth = torch.ones(n_rays, dtype=type, device=device) * min_depth #将最小深度复制n_rays份
    
    if not torch.is_tensor(max_depth):
        max_depth = torch.ones(n_rays, dtype=type, device=device) * max_depth #将最大深度复制n_rays份

    depth_range = max_depth - min_depth #深度范围
  
    lower_limits_scale = depth_range[..., None] * bin_limits_scale + min_depth[..., None] #每个bin的深度范围
    lower_limits_scale = lower_limits_scale[:, :-1] #去掉最后一个bin，因为最后一个bin的上限是max_depth

    assert lower_limits_scale.shape == (n_rays, n_bins)

    bin_length_scale = depth_range / n_bins  #每个bin的深度范围
    increments_scale = torch.rand(
        n_rays, n_bins, device=device,
        dtype=torch.float32) * bin_length_scale[..., None] #每个bin的深度范围内随机采样

    z_vals_scale = lower_limits_scale + increments_scale #在均匀采样的周围随机采样，说白了还是随机采样

    assert z_vals_scale.shape == (n_rays, n_bins)

    return z_vals_scale

# @torch.jit.script
def normal_bins_sampling(depth, n_bins, n_rays, delta, device = "cuda:0"):
    # type: (Tensor, int, int, float) -> Tensor

    # device = "cpu"
    # bins = torch.normal(0.0, delta / 3., size=[n_rays, n_bins], devi
        # self.keyframes_batch = torch.empty(self.n_keyframes,ce=device).sort().values
    bins = torch.empty(n_rays, n_bins, dtype=torch.float32, device=device).normal_(mean=0.,std=delta / 3.).sort().values
    bins = torch.clip(bins, -delta, delta)
    z_vals = depth[:, None] + bins

    assert z_vals.shape == (n_rays, n_bins)

    return z_vals  #论文中的normal采样


class sceneObject:
    """
    object instance mapping,
    updating keyframes, get training samples, optimizing MLP map
    目标实例建图，更新关键帧，获取训练样本，优化MLP地图
    """

    def __init__(self, cfg, obj_id, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor, live_frame_id) -> None:
        self.do_bg = cfg.do_bg
        self.obj_id = obj_id
        self.data_device = cfg.data_device
        self.training_device = cfg.training_device

        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)

        if self.do_bg and self.obj_id == 0: # do seperate bg，背景单独建图
            self.obj_scale = cfg.bg_scale #5.0，背景的缩放比例
            self.hidden_feature_size = cfg.hidden_feature_size_bg #128，背景特征隐藏层大小
            self.n_bins_cam2surface = cfg.n_bins_cam2surface_bg #1，相机到表面的bin的数量？
            self.keyframe_step = cfg.keyframe_step_bg #50，背景关键帧的步长
        else: #物体
            self.obj_scale = cfg.obj_scale #2.0，物体的缩放比例
            self.hidden_feature_size = cfg.hidden_feature_size #32，物体特征隐藏层大小
            self.n_bins_cam2surface = cfg.n_bins_cam2surface
            self.keyframe_step = cfg.keyframe_step #25，关键帧的步长

        self.frames_width = rgb.shape[0] #680，相机的宽度减去margin
        self.frames_height = rgb.shape[1] #680，相机的高度减去margin

        self.min_bound = cfg.min_depth #0.0，最小深度是0米
        self.max_bound = cfg.max_depth #8.0，最大深度是8米，因此只适合室内场景
        self.n_bins = cfg.n_bins #9，bin的数量
        self.n_unidir_funcs = cfg.n_unidir_funcs #10，单向函数的数量？

        self.surface_eps = cfg.surface_eps #0.1，表面的eps
        self.stop_eps = cfg.stop_eps #0.015，其他的eps

        self.n_keyframes = 1  # Number of keyframes，初始化关键帧数量为1
        self.kf_pointer = None # 关键帧的指针，一般用于存储关键帧数量
        self.keyframe_buffer_size = cfg.keyframe_buffer_size #20，关键帧的缓存大小
        self.kf_id_dict = bidict({live_frame_id:0})
        self.kf_buffer_full = False #关键帧缓存是否已满
        self.frame_cnt = 0  # number of frames taken in，已经采集的帧数
        self.lastest_kf_queue = [] #最新的关键帧队列

        self.bbox = torch.empty(  # obj bounding bounding box in the frame
            self.keyframe_buffer_size,
            4,
            device=self.data_device)  # [u low, u high, v low, v high]
        self.bbox[0] = bbox_2d

        # RGB + pixel state batch，RGB和像素状态批处理
        self.rgb_idx = slice(0, 3) # RGB的索引
        self.state_idx = slice(3, 4)  # pixel state的索引
        self.rgbs_batch = torch.empty(self.keyframe_buffer_size,
                                      self.frames_width,
                                      self.frames_height,
                                      4,
                                      dtype=torch.uint8,
                                      device=self.data_device)

        # Pixel states:
        self.other_obj = 0  # pixel doesn't belong to obj，像素不属于obj
        self.this_obj = 1  # pixel belong to obj，像素属于obj
        self.unknown_obj = 2  # pixel state is unknown，像素状态未知

        # Initialize first frame rgb and pixel state
        self.rgbs_batch[0, :, :, self.rgb_idx] = rgb
        self.rgbs_batch[0, :, :, self.state_idx] = mask[..., None] #mask是像素状态

        self.depth_batch = torch.empty(self.keyframe_buffer_size,
                                       self.frames_width,
                                       self.frames_height,
                                       dtype=torch.float32,
                                       device=self.data_device)

        # Initialize first frame's depth ，初始化第一帧的深度
        self.depth_batch[0] = depth
        self.t_wc_batch = torch.empty(
            self.keyframe_buffer_size, 4, 4,
            dtype=torch.float32,
            device=self.data_device)  # world to camera transform，世界坐标到相机坐标的变换

        # Initialize first frame's world2cam transform，初始化第一帧的世界坐标到相机坐标的变换
        self.t_wc_batch[0] = t_wc

        # neural field map，神经场地图
        trainer_cfg = copy.deepcopy(cfg)
        trainer_cfg.obj_id = self.obj_id
        trainer_cfg.hidden_feature_size = self.hidden_feature_size
        trainer_cfg.obj_scale = self.obj_scale #2.0，物体的缩放比例
        self.trainer = trainer.Trainer(trainer_cfg) #初始化训练器

        # 3D boundary
        self.bbox3d = None
        self.pc = []

        # init  obj local frame
        # self.obj_center = self.init_obj_center(intrinsic, depth, mask, t_wc)
        self.obj_center = torch.tensor(0.0) # shouldn't make any difference because of frequency embedding


    def init_obj_center(self, intrinsic_open3d, depth, mask, t_wc): #初始化物体中心
        obj_depth = depth.cpu().clone() #深度图
        obj_depth[mask!=self.this_obj] = 0
        T_CW = np.linalg.inv(t_wc.cpu().numpy()) #世界坐标到相机坐标的变换，=相机坐标到世界坐标的变换的逆
        pc_obj_init = open3d.geometry.PointCloud.create_from_depth_image(
            depth=open3d.geometry.Image(np.asarray(obj_depth.permute(1,0).numpy(), order="C")),
            intrinsic=intrinsic_open3d,
            extrinsic=T_CW,
            depth_trunc=self.max_bound,
            depth_scale=1.0)  #让open3d自己计算深度图,深度图的范围是[0,8]米，深度图的缩放比例是1.0
        obj_center = torch.from_numpy(np.mean(pc_obj_init.points, axis=0)).float()
        return obj_center

    # @profile
    #添加关键帧
    def append_keyframe(self, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor, frame_id:np.uint8=1):
        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)
        assert self.n_keyframes <= self.keyframe_buffer_size - 1
        assert rgb.dtype == torch.uint8
        assert mask.dtype == torch.uint8
        assert depth.dtype == torch.float32

        # every kf_step choose one kf
        is_kf = (self.frame_cnt % self.keyframe_step == 0) or self.n_keyframes == 1 #每隔kf_step帧选择一个关键帧
        # print("---------------------")
        # print("self.kf_id_dict ", self.kf_id_dict)
        # print("live frame id ", frame_id)
        # print("n_frames ", self.n_keyframes)
        if self.n_keyframes == self.keyframe_buffer_size - 1:  # 关键帧缓存已满，需要清理
            self.kf_buffer_full = True
            if self.kf_pointer is None:
                self.kf_pointer = self.n_keyframes  # kf pointer指向最老的关键帧
            #处理较旧的关键帧
            self.rgbs_batch[self.kf_pointer, :, :, self.rgb_idx] = rgb #存储rgb
            self.rgbs_batch[self.kf_pointer, :, :, self.state_idx] = mask[..., None] #存储像素状态（0,1,2）
            self.depth_batch[self.kf_pointer, ...] = depth #储存深度图
            self.t_wc_batch[self.kf_pointer, ...] = t_wc #存储位姿（世界坐标到相机坐标的变换）
            self.bbox[self.kf_pointer, ...] = bbox_2d #存储2D边界框
            self.kf_id_dict.inv[self.kf_pointer] = frame_id #存储关键帧id

            if is_kf:
                self.lastest_kf_queue.append(self.kf_pointer) #添加关键帧到队列
                pruned_frame_id, pruned_kf_id = self.prune_keyframe() #使用简单的随机选择策略清理关键帧，不清理最后两帧
                self.kf_pointer = pruned_kf_id
                print("pruned kf id ", self.kf_pointer)

        else: # 关键帧缓存未满，直接添加
            if not is_kf:   # not kf, replace，不是关键帧，替换
                self.rgbs_batch[self.n_keyframes-1, :, :, self.rgb_idx] = rgb
                self.rgbs_batch[self.n_keyframes-1, :, :, self.state_idx] = mask[..., None]
                self.depth_batch[self.n_keyframes-1, ...] = depth
                self.t_wc_batch[self.n_keyframes-1, ...] = t_wc
                self.bbox[self.n_keyframes-1, ...] = bbox_2d
                self.kf_id_dict.inv[self.n_keyframes-1] = frame_id #存储关键帧id，.inv是反转字典,反转字典的作用是可以通过value找到key
            else:   # is kf, add new kf，是关键帧，添加新的关键帧
                self.kf_id_dict[frame_id] = self.n_keyframes
                self.rgbs_batch[self.n_keyframes, :, :, self.rgb_idx] = rgb
                self.rgbs_batch[self.n_keyframes, :, :, self.state_idx] = mask[..., None]
                self.depth_batch[self.n_keyframes, ...] = depth
                self.t_wc_batch[self.n_keyframes, ...] = t_wc
                self.bbox[self.n_keyframes, ...] = bbox_2d
                self.lastest_kf_queue.append(self.n_keyframes) #添加关键帧到队列
                self.n_keyframes += 1 #关键帧数量加1

        # print("self.kf_id_dic ", self.kf_id_dict)
        self.frame_cnt += 1 #已经采集的帧数加1
        if len(self.lastest_kf_queue) > 2:  # keep latest two frames，保留最后两帧
            self.lastest_kf_queue = self.lastest_kf_queue[-2:]

    def prune_keyframe(self): #清理关键帧
        # simple strategy to prune, randomly choose，简单的策略，随机选择
        key, value = random.choice(list(self.kf_id_dict.items())[:-2])  # do not prune latest two frames，不要清理最后两帧
        return key, value

    def get_bound(self, intrinsic_open3d): #利用open3d和trimesh获取3D边界
        # get 3D boundary from posed depth img   todo update sparse pc when append frame，从深度图中获取3D边界，为了更新稀疏点云
        pcs = open3d.geometry.PointCloud()
        for kf_id in range(self.n_keyframes): #遍历所有关键帧
            mask = self.rgbs_batch[kf_id, :, :, self.state_idx].squeeze() == self.this_obj #像素状态为1的像素属于obj，制作掩码mask
            depth = self.depth_batch[kf_id].cpu().clone()
            twc = self.t_wc_batch[kf_id].cpu().numpy() #位姿，世界坐标到相机坐标的变换
            depth[~mask] = 0 #深度图中像素状态不为1的像素深度置为0
            depth = depth.permute(1,0).numpy().astype(np.float32)
            T_CW = np.linalg.inv(twc) #位姿相机坐标到世界坐标的变换
            pc = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(np.asarray(depth, order="C")), intrinsic=intrinsic_open3d, extrinsic=T_CW)
            #让open3d自己计算深度图，即从深度图中获取点云，深度图的范围是[0,8]米，深度图的缩放比例是1.0
            # self.pc += pc
            pcs += pc #将所有关键帧的点云合并
            #接下来就将合并后的点云交给open3d和trimesh进行最小边界框的计算

        # # get minimal oriented 3d bbox，获取最小的3D边界框
        # try:
        #     bbox3d = open3d.geometry.OrientedBoundingBox.create_from_points(pcs.points)
        # except RuntimeError:
        #     print("too few pcs obj ")
        #     return None
        # trimesh has a better minimal bbox implementation than open3d
        try:
            transform, extents = trimesh.bounds.oriented_bounds(np.array(pcs.points))  # pc
            transform = np.linalg.inv(transform)
        except scipy.spatial._qhull.QhullError:
            print("too few pcs obj ") #点云太少
            return None

        for i in range(extents.shape[0]):
            extents[i] = np.maximum(extents[i], 0.10)  # at least rendering 10cm
        bbox = utils.BoundingBox()
        bbox.center = transform[:3, 3]
        bbox.R = transform[:3, :3]
        bbox.extent = extents
        bbox3d = open3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)

        min_extent = 0.05
        bbox3d.extent = np.maximum(min_extent, bbox3d.extent)
        bbox3d.color = (255,0,0)
        self.bbox3d = utils.bbox_open3d2bbox(bbox_o3d=bbox3d)
        # self.pc = []
        print("obj ", self.obj_id)
        print("bound ", bbox3d)
        print("kf id dict ", self.kf_id_dict)
        # open3d.visualization.draw_geometries([bbox3d, pcs])
        return bbox3d



    def get_training_samples(self, n_frames, n_samples, cached_rays_dir): #采样，
        # Sample pixels
        # 首先要确保最新的两帧被采样到
        if self.n_keyframes > 2: # make sure latest 2 frames are sampled    todo if kf pruned, this is not the latest frame
            keyframe_ids = torch.randint(low=0,
                                         high=self.n_keyframes,
                                         size=(n_frames - 2,),
                                         dtype=torch.long,
                                         device=self.data_device) #随机选择n_frames-2个关键帧
            # if self.kf_buffer_full:
            # latest_frame_ids = list(self.kf_id_dict.values())[-2:]
            latest_frame_ids = self.lastest_kf_queue[-2:] #最新的两帧
            keyframe_ids = torch.cat([keyframe_ids,
                                          torch.tensor(latest_frame_ids, device=keyframe_ids.device)]) #将最新的两帧添加到随机选择的采样关键帧中
            # print("latest_frame_ids", latest_frame_ids)
            # else:   # sample last 2 frames
            #     keyframe_ids = torch.cat([keyframe_ids,
            #                               torch.tensor([self.n_keyframes-2, self.n_keyframes-1], device=keyframe_ids.device)])
        else:
            keyframe_ids = torch.randint(low=0,
                                         high=self.n_keyframes,
                                         size=(n_frames,),
                                         dtype=torch.long,
                                         device=self.data_device)
        keyframe_ids = torch.unsqueeze(keyframe_ids, dim=-1)
        idx_w = torch.rand(n_frames, n_samples, device=self.data_device) #随机选择n_samples个像素，idx_w是像素的水平索引
        idx_h = torch.rand(n_frames, n_samples, device=self.data_device) #idx_h是像素的垂直索引

        # resizing idx_w and idx_h to be in the bbox range, 将idx_w和idx_h调整到bbox范围内
        idx_w = idx_w * (self.bbox[keyframe_ids, 1] - self.bbox[keyframe_ids, 0]) + self.bbox[keyframe_ids, 0]
        idx_h = idx_h * (self.bbox[keyframe_ids, 3] - self.bbox[keyframe_ids, 2]) + self.bbox[keyframe_ids, 2]

        idx_w = idx_w.long()
        idx_h = idx_h.long()

        sampled_rgbs = self.rgbs_batch[keyframe_ids, idx_w, idx_h] #采样的rgb
        sampled_depth = self.depth_batch[keyframe_ids, idx_w, idx_h] #采样的深度

        # Get ray directions for sampled pixels，
        sampled_ray_dirs = cached_rays_dir[idx_w, idx_h] #采样的射线方向

        # Get sampled keyframe poses, 获取采样关键帧的位姿
        sampled_twc = self.t_wc_batch[keyframe_ids[:, 0], :, :]

        origins, dirs_w = origin_dirs_W(sampled_twc, sampled_ray_dirs) #获取采样射线的世界坐标系下的方向和原点

        return self.sample_3d_points(sampled_rgbs, sampled_depth, origins, dirs_w) #根据采样的rgb和深度，以及射线的方向和原点，获取3D采样点

    # todo 终于轮到论文里比较重要的3D采样策略了
    def sample_3d_points(self, sampled_rgbs, sampled_depth, origins, dirs_w): #获取3D采样点
        """
        3D sampling strategy
        论文里的3D采样策略
        * For pixels with invalid depth:（状态=0）
            - N+M from minimum bound to max (stratified)
        对于无效深度的像素： - N+M从最小边界到最大（均匀采样）
        * For pixels with valid depth: 对于有效深度的像素：
            # Pixel belongs to this object 如果像素属于这个对象（状态=1）
                - N from cam to surface (stratified) 从相机到表面均匀采样N个点
                - M around surface (stratified/normal) 在表面周围（均匀/mormal）
            # Pixel belongs that don't belong to this object 如果像素不属于这个对象
                - N from cam to surface (stratified) 从相机到表面均匀采样N个点
                - M around surface (stratified) 在表面周围均为采样M个点
            # Pixel with unknown state 如果像素状态未知(状态=2)
                - Do nothing! 不做任何事情
        """

        n_bins_cam2surface = self.n_bins_cam2surface #1，相机到表面的采样点的数量
        n_bins = self.n_bins #9，表面周围采样点bin的数量
        eps = self.surface_eps #0.1，表面的eps
        other_objs_max_eps = self.stop_eps #0.05   # todo 0.02
        # print("max depth ", torch.max(sampled_depth))
        sampled_z = torch.zeros(
            sampled_rgbs.shape[0] * sampled_rgbs.shape[1],
            n_bins_cam2surface + n_bins,
            dtype=self.depth_batch.dtype,
            device=self.data_device)  # shape (N*n_rays, n_bins_cam2surface + n_bins)

        invalid_depth_mask = (sampled_depth <= self.min_bound).view(-1) #深度小于最小深度的像素，均为无效深度
        # max_bound = self.max_bound
        max_bound = torch.max(sampled_depth) #最大深度
        # sampling for points with invalid depth
        invalid_depth_count = invalid_depth_mask.count_nonzero() #无效深度的像素数量
        if invalid_depth_count:
            sampled_z[invalid_depth_mask, :] = stratified_bins(
                self.min_bound, max_bound,
                n_bins_cam2surface + n_bins, invalid_depth_count,
                device=self.data_device)  # 对于无效深度的像素： - N+M从最小边界到最大（均匀采样）

        # sampling for valid depth rays
        valid_depth_mask = ~invalid_depth_mask
        valid_depth_count = valid_depth_mask.count_nonzero()

        # 对于有效深度的像素：
        if valid_depth_count:
            # Sample between min bound and depth for all pixels with valid depth，对于所有有效深度的像素，在最小边界和深度之间采样
            sampled_z[valid_depth_mask, :n_bins_cam2surface] = stratified_bins(
                self.min_bound, sampled_depth.view(-1)[valid_depth_mask]-eps,
                n_bins_cam2surface, valid_depth_count, device=self.data_device) #从最小边界到深度均匀采样n_bins_cam2surface个点

            # sampling around depth for this object
            obj_mask = (sampled_rgbs[..., -1] == self.this_obj).view(-1) & valid_depth_mask  #在物体表面周围采样
            assert sampled_z.shape[0] == obj_mask.shape[0]
            obj_count = obj_mask.count_nonzero()

            if obj_count:
                sampling_method = "normal"  # stratified or normal，均匀采样或normal采样
                if sampling_method == "stratified":
                    sampled_z[obj_mask, n_bins_cam2surface:] = stratified_bins(
                        sampled_depth.view(-1)[obj_mask] - eps, sampled_depth.view(-1)[obj_mask] + eps,
                        n_bins, obj_count, device=self.data_device)

                elif sampling_method == "normal":
                    sampled_z[obj_mask, n_bins_cam2surface:] = normal_bins_sampling(
                        sampled_depth.view(-1)[obj_mask],
                        n_bins,
                        obj_count,
                        delta=eps,
                        device=self.data_device)

                else:
                    raise (
                        f"sampling method not implemented {sampling_method}, \
                            stratified and normal sampling only currenty implemented."
                    )

            # sampling around depth of other objects
            other_obj_mask = (sampled_rgbs[..., -1] != self.this_obj).view(-1) & valid_depth_mask
            other_objs_count = other_obj_mask.count_nonzero()
            if other_objs_count:
                sampled_z[other_obj_mask, n_bins_cam2surface:] = stratified_bins(
                    sampled_depth.view(-1)[other_obj_mask] - eps,
                    sampled_depth.view(-1)[other_obj_mask] + other_objs_max_eps,
                    n_bins, other_objs_count, device=self.data_device)

        sampled_z = sampled_z.view(sampled_rgbs.shape[0],
                                   sampled_rgbs.shape[1],
                                   -1)  # view as (n_rays, n_samples, 10)
        input_pcs = origins[..., None, None, :] + (dirs_w[:, :, None, :] *
                                                   sampled_z[..., None])
        input_pcs -= self.obj_center
        obj_labels = sampled_rgbs[..., -1].view(-1)
        return sampled_rgbs[..., :3], sampled_depth, valid_depth_mask, obj_labels, input_pcs, sampled_z
        #返回采样的rgb、深度、有效深度的掩码、像素状态、输入点云、采样的深度

    def save_checkpoints(self, path, epoch):
        obj_id = self.obj_id
        chechpoint_load_file = (path + "/obj_" + str(obj_id) + "_frame_" + str(epoch) + ".pth")
        # 把checkpoint保存到文件中
        torch.save(
            {
                "epoch": epoch,
                "FC_state_dict": self.trainer.fc_occ_map.state_dict(),
                "PE_state_dict": self.trainer.pe.state_dict(),
                "obj_id": self.obj_id,
                "bbox": self.bbox3d,
                "obj_scale": self.trainer.obj_scale
            },
            chechpoint_load_file,
        )
        # optimiser?

    def load_checkpoints(self, ckpt_file): #加载checkpoint
        checkpoint_load_file = (ckpt_file)
        if not os.path.exists(checkpoint_load_file):
            print("ckpt not exist ", checkpoint_load_file)
            return
        checkpoint = torch.load(checkpoint_load_file)
        self.trainer.fc_occ_map.load_state_dict(checkpoint["FC_state_dict"])
        self.trainer.pe.load_state_dict(checkpoint["PE_state_dict"])
        self.obj_id = checkpoint["obj_id"]
        self.bbox3d = checkpoint["bbox"]
        self.trainer.obj_scale = checkpoint["obj_scale"]

        self.trainer.fc_occ_map.to(self.training_device)
        self.trainer.pe.to(self.training_device)


class cameraInfo:

    def __init__(self, cfg) -> None:
        self.device = cfg.data_device
        self.width = cfg.W  # Frame width
        self.height = cfg.H  # Frame height

        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy

        self.rays_dir_cache = self.get_rays_dirs()

    def get_rays_dirs(self, depth_type="z"): #获取射线方向
        idx_w = torch.arange(end=self.width, device=self.device) #水平索引
        idx_h = torch.arange(end=self.height, device=self.device) #垂直索引

        dirs = torch.ones((self.width, self.height, 3), device=self.device)

        dirs[:, :, 0] = ((idx_w - self.cx) / self.fx)[:, None] #射线方向的x分量
        dirs[:, :, 1] = ((idx_h - self.cy) / self.fy) #射线方向的y分量

        if depth_type == "euclidean":
            raise Exception(
                "Get camera rays directions with euclidean depth not yet implemented"
            )
            norm = torch.norm(dirs, dim=-1)
            dirs = dirs * (1. / norm)[:, :, :, None]

        return dirs

