# 文件路径: pcdet/models/detectors/esformer.py

import torch
import torch.nn as nn
from .detector3d_template import Detector3DTemplate


class ESFormer(Detector3DTemplate):
    """
    ESFormer: 基于 Swin Transformer 的 3D 目标检测器
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """
        前向传播
        Args:
            batch_dict: 包含点云数据和其他信息的字典
        Returns:
            batch_dict: 更新后的字典，包含预测结果
        """
        # 按照模块拓扑顺序进行前向传播
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # 训练模式下计算损失
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # 测试模式下进行后处理
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        """
        计算训练损失
        """
        disp_dict = {}
        loss = 0

        # 从 dense_head 获取损失
        if hasattr(self, 'dense_head') and self.dense_head is not None:
            loss_dense, tb_dict_dense = self.dense_head.get_loss()
            loss += loss_dense
            disp_dict.update(tb_dict_dense)

        # 从 point_head 获取损失（如果存在）
        if hasattr(self, 'point_head') and self.point_head is not None:
            loss_point, tb_dict_point = self.point_head.get_loss()
            loss += loss_point
            for key, val in tb_dict_point.items():
                disp_dict['point_' + key] = val

        # 从 roi_head 获取损失（如果存在）
        if hasattr(self, 'roi_head') and self.roi_head is not None:
            loss_roi, tb_dict_roi = self.roi_head.get_loss()
            loss += loss_roi
            for key, val in tb_dict_roi.items():
                disp_dict['roi_' + key] = val

        tb_dict = {
            'loss': loss.item(),
            **disp_dict
        }

        return loss, tb_dict, disp_dict

    def build_pces_tool(self, model_info_dict):
        """
        构建 PCES (Point Cloud Enhancement Strategy) 工具
        """
        if self.model_cfg.get('PCES_TOOL', None) is None:
            return None, model_info_dict

        from ..model_utils import PCESTool
        
        pces_tool = PCESTool(
            model_cfg=self.model_cfg.PCES_TOOL,
            input_channels=model_info_dict['num_rawpoint_features']
        )
        
        model_info_dict['module_list'].append(pces_tool)
        return pces_tool, model_info_dict