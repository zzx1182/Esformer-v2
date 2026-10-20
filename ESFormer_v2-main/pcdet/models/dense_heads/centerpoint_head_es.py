# /workspace/OpenPCDet/pcdet/models/dense_heads/centerpoint_head_es.py
# ==============================================================================
#                      !!! 开始复制这里的全部代码 !!!
# ==============================================================================
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from .center_head import CenterHead


class CenterPointHeadES(CenterHead):
    """
    这是一个针对ESFormer定制的CenterPoint检测头。
    关键修改在于 __init__ 函数的定义，以兼容新版OpenPCDet框架传递的额外参数。
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 voxel_size, predict_boxes_when_training=True, **kwargs): # <--- 关键修改在这里
        """
        Args:
            model_cfg:           模型配置
            input_channels:      输入通道数
            num_class:           类别数量
            class_names:         类别名称
            grid_size:           网格尺寸
            point_cloud_range:   点云范围
            voxel_size:          体素大小 (新增的参数)
            predict_boxes_when_training: 是否在训练时预测边界框
            **kwargs:            用于接收所有其他意料之外的参数
        """
        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,  # <--- 将接收到的参数传递给父类
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.forward_ret_dict = {}

    def assign_targets(self, gt_boxes):
        """
        重写此方法以适应您可能有的特定目标分配逻辑，
        如果与父类 CenterHead 相同，则此函数可以为空或直接调用super()。
        这里的实现保持与标准CenterHead一致。
        """
        return super().assign_targets(gt_boxes)

    def forward(self, data_dict):
        """
        前向传播函数。
        """
        # 调用父类的前向传播，这是标准做法
        # 这会填充 self.forward_ret_dict
        super().forward(data_dict)

        if not self.training:
            # 在推理或测试时，进行后处理以生成最终的边界框
            pred_dicts, recall_dicts = self.post_processing(data_dict)
            return pred_dicts, recall_dicts

        # 在训练时，返回包含损失和中间结果的字典
        return self.forward_ret_dict

# ==============================================================================
#                      !!! 在这里结束复制 !!!
# ==============================================================================