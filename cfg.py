import json
import numpy as np
import os
import utils

class Config:
    def __init__(self, config_file):
        # setting params
        with open(config_file) as json_file:
            config = json.load(json_file)

        # training strategy
        self.do_bg = bool(config["trainer"]["do_bg"]) # 肯定都是有背景的，1
        self.training_device = config["trainer"]["train_device"] # cuda:0
        self.data_device = config["trainer"]["data_device"] # cuda:0
        self.max_n_models = config["trainer"]["n_models"] # 看你机器的内存了，我这里是10，默认是100
        self.live_mode = bool(config["dataset"]["live"]) # 0，直播网格的可视化
        self.keep_live_time = config["dataset"]["keep_alive"]  # 20
        self.imap_mode = config["trainer"]["imap_mode"] # 0
        self.training_strategy = config["trainer"]["training_strategy"]  # "forloop" "vmap"
        self.obj_id = -1 #初始化obj_id为-1

        # dataset setting
        self.dataset_format = config["dataset"]["format"] # "Replica"
        self.dataset_dir = config["dataset"]["path"] # "data/Replica"
        self.depth_scale = 1 / config["trainer"]["scale"] # 深度的缩放比例，1/1000，因为深度图是毫米级别的，而我们的模型是米级别的
        # camera setting
        self.max_depth = config["render"]["depth_range"][1] # [0.0, 8.0]，最大深度是8米，因此只适合室内场景
        self.min_depth = config["render"]["depth_range"][0] # [0.0, 8.0]，最小深度是0米
        self.mh = config["camera"]["mh"] # 0，
        self.mw = config["camera"]["mw"] # 0，是什么意思呢？是相机的margin，因为相机的视野是有限的，因此需要margin
        self.height = config["camera"]["h"] # 680，相机的高度
        self.width = config["camera"]["w"] # 680，相机的宽度
        self.H = self.height - 2 * self.mh # 680，相机的高度减去margin
        self.W = self.width - 2 * self.mw # 680，相机的宽度减去margin
        if "fx" in config["camera"]:
            self.fx = config["camera"]["fx"] # 600.0，相机的焦距，fx是相机的焦距，fy是相机的焦距，cx是相机的中心点，cy是相机的中心点
            self.fy = config["camera"]["fy"] # 600.0，相机的焦距，一般来说fx=fy
            self.cx = config["camera"]["cx"] - self.mw # 599.5，相机的中心点
            self.cy = config["camera"]["cy"] - self.mh # 339.5
        else:   # for scannet,要从文件中读取相机的内参
            intrinsic = utils.load_matrix_from_txt(os.path.join(self.dataset_dir, "intrinsic/intrinsic_depth.txt"))
            self.fx = intrinsic[0, 0]
            self.fy = intrinsic[1, 1]
            self.cx = intrinsic[0, 2] - self.mw
            self.cy = intrinsic[1, 2] - self.mh
        if "distortion" in config["camera"]: # 畸变参数
            self.distortion_array = np.array(config["camera"]["distortion"])
        elif "k1" in config["camera"]:
            k1 = config["camera"]["k1"]
            k2 = config["camera"]["k2"]
            k3 = config["camera"]["k3"]
            k4 = config["camera"]["k4"]
            k5 = config["camera"]["k5"]
            k6 = config["camera"]["k6"]
            p1 = config["camera"]["p1"]
            p2 = config["camera"]["p2"]
            self.distortion_array = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        else:
            self.distortion_array = None

        # training setting
        self.win_size = config["model"]["window_size"] # 5，滑动窗口大小
        self.n_iter_per_frame = config["render"]["iters_per_frame"] # 20，每一帧的迭代次数
        self.n_per_optim = config["render"]["n_per_optim"] # 120，每一次优化的点的数量
        self.n_samples_per_frame = self.n_per_optim // self.win_size # 24，每一帧的采样点的数量
        self.win_size_bg = config["model"]["window_size_bg"] # 10，背景的滑动窗口大小
        self.n_per_optim_bg = config["render"]["n_per_optim_bg"] # 1200，背景每一次优化的点的数量
        self.n_samples_per_frame_bg = self.n_per_optim_bg // self.win_size_bg # 120，背景每一帧的采样点的数量
        self.keyframe_buffer_size = config["model"]["keyframe_buffer_size"] # 20，关键帧的缓存大小
        self.keyframe_step = config["model"]["keyframe_step"] # 25，关键帧的步长
        self.keyframe_step_bg = config["model"]["keyframe_step_bg"] # 50，背景关键帧的步长
        self.obj_scale = config["model"]["obj_scale"] # 2.0，物体的缩放比例
        self.bg_scale = config["model"]["bg_scale"] # 5.0，背景的缩放比例
        self.hidden_feature_size = config["model"]["hidden_feature_size"] # 32，物体特征隐藏层大小
        self.hidden_feature_size_bg = config["model"]["hidden_feature_size_bg"] # 128，背景特征隐藏层大小
        self.n_bins_cam2surface = config["render"]["n_bins_cam2surface"] # 1，相机到表面的bin的数量？
        self.n_bins_cam2surface_bg = config["render"]["n_bins_cam2surface_bg"] # 5
        self.n_bins = config["render"]["n_bins"] # 9，bin的数量
        self.n_unidir_funcs = config["model"]["n_unidir_funcs"] # 10，单向函数的数量？
        self.surface_eps = config["model"]["surface_eps"] # 0.1，表面的eps
        self.stop_eps = config["model"]["other_eps"] # 0.015，其他的eps

        # optimizer setting
        self.learning_rate = config["optimizer"]["args"]["lr"] # 0.001
        self.weight_decay = config["optimizer"]["args"]["weight_decay"] # 0.013

        # vis setting
        self.vis_device = config["vis"]["vis_device"] # cuda:0
        self.n_vis_iter = config["vis"]["n_vis_iter"] # 500，可视化的迭代次数
        self.live_voxel_size = config["vis"]["live_voxel_size"] # 0.005
        self.grid_dim = config["vis"]["grid_dim"] # 128，网格的维度