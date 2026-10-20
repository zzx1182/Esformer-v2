import copy
import pickle
from pathlib import Path

import numpy as np
from skimage import io

# from . import kitti_utils  # 根据您的需要，有些功能可能不再需要
# from ...ops.roiaware_pool3d import roiaware_pool3d_utils # GT Database创建时才需要
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    # ==================================================================================================================
    # 以下获取 Image, Label, Calib 等函数在仅有LiDAR数据时不再被直接调用，保留原始定义以防万一
    # ==================================================================================================================
    def get_image_shape(self, idx):
        # 如果需要，可以返回一个假的形状
        return np.array([0, 0], dtype=np.int32)

    def get_label(self, idx):
        # 如果没有label文件，返回空列表
        return []

    def get_calib(self, idx):
        # 如果没有calib文件，返回None或者一个空的标定对象
        return None

    # ==================================================================================================================
    # 核心修改区域: process_single_scene 函数
    # ==================================================================================================================
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            """
            处理数据集中的单个场景（即单个点云文件）。
            由于我们假设只有点云数据，这个函数会加载真实的点云，
            并为所有其他数据（如图像、标注等）创建空的或假的占位符。
            """
            # (1) 打印当前处理的样本ID，方便调试
            print('%s sample_idx: %s' % (self.split, sample_idx))
            
            info = {}

            # (2) 加载我们唯一拥有的真实数据：点云
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # (3) 为所有缺失的数据创建假的“占位符”，以确保数据结构完整
            # 创建一个假的 'image' 字典
            info['image'] = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}

            # 创建一个空的 'calib' (标定) 字典
            info['calib'] = {}

            # (4) 【关键】如果has_label为False或没有label文件，则创建空的标注字典
            # 这样可以完全跳过所有处理Label的代码，避免错误
            if has_label:
                # 即使has_label为True，我们也假设没有label文件，因此创建空标注
                info['annos'] = {
                    'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                    'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([]),
                    'difficulty': np.array([]), 'index': np.array([]),
                    'gt_boxes_lidar': np.zeros([0, 7], dtype=np.float32)
                }
            
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    # 由于没有GT Labels, create_groundtruth_database 无法执行, 保留函数定义但确保它不被调用
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        if self.logger:
            self.logger.warning("Attempted to create groundtruth database, but this is not supported for point-cloud-only data. Skipping.")
        return

    # __getitem__也需要修改以适应没有标注的情况
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']

        # 初始化输入字典
        input_dict = {
            'frame_id': sample_idx,
        }

        # 加载点云数据 (这是我们唯一有的真实数据)
        points = self.get_lidar(sample_idx)
        input_dict['points'] = points

        # 由于我们没有标注，所以创建一个空的gt_boxes来保证数据格式的完整性
        # 这对于许多模型是必需的，即使在推理时也是如此
        input_dict['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
        input_dict['gt_names'] = np.array([])
        
        # 即使没有图像，也提供一个假的形状，因为某些后续处理可能会用到
        input_dict['image_shape'] = info['image']['image_shape']

        # 调用prepare_data进行数据增强等操作
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    # 其他函数保留原样，因为它们主要用于评估和预测，在数据准备阶段不会被调用

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        # 此函数在推理时使用，依赖标定，如果只进行训练则无需修改
        pass

    def evaluation(self, det_annos, class_names, **kwargs):
        # 评估需要真值，对于只有点云的数据集，无法进行评估
        if 'annos' not in self.kitti_infos[0].keys() or len(self.kitti_infos[0]['annos']['name']) == 0:
            if self.logger:
                self.logger.warning("Evaluation skipped since no ground truth annotations were found.")
            return None, {}
        
        from .kitti_object_eval_python import eval as kitti_eval
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict
    
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    # has_label 设置为 False，因为我们没有标注文件
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    # ==================================================================================================================
    # 核心修改区域: 移除 create_groundtruth_database 的调用
    # ==================================================================================================================
    print('---------------Skipping create groundtruth database (not applicable for point-cloud-only data)---------------')
    # dataset.set_split(train_split)
    # dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'], # 您可以根据需要修改这里的类别
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )