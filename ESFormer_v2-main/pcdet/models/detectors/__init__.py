# 文件路径: pcdet/models/detectors/__init__.py

from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .point_rcnn import PointRCNN
from .second_net import SECONDNet  # 修正: 使用 second_net 而不是 second
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn import PVRCNN
from .PartA2_net import PartA2Net  # 修正: 使用正确的类名
from .esformer import ESFormer

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,  # 修正: 使用正确的类名
    'PartA2Net': PartA2Net,  # 修正: 使用正确的类名
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'ESFormer': ESFormer
}


def build_detector(model_cfg, num_class, dataset):
    """
    构建检测器模型
    Args:
        model_cfg: 模型配置
        num_class: 类别数量
        dataset: 数据集对象
    Returns:
        model: 构建的检测器模型
    """
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model