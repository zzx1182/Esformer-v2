# /workspace/OpenPCDet/pcdet/models/dense_heads/__init__.py

from .anchor_head_single import AnchorHeadSingle
from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
# 下面这行是您的 ESFormer 模型特别需要的
from .centerpoint_head_es import CenterPointHeadES

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    # 确保您的模型被注册
    'CenterPointHeadES': CenterPointHeadES
}