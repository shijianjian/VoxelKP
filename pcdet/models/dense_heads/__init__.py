from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .center_head_kp import CenterHeadKP
from .center_head_kp_naive import CenterHeadKPNaive
from .center_head_kp_direct import CenterHeadKPDirect
from .voxelnext_head import VoxelNeXtHead
from .voxelnext_head_kp_merge import VoxelNeXtHeadKPMerge
from .transfusion_head import TransFusionHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'CenterHeadKP': CenterHeadKP,
    'CenterHeadKPNaive': CenterHeadKPNaive,
    'CenterHeadKPDirect': CenterHeadKPDirect,
    'VoxelNeXtHead': VoxelNeXtHead,
    'VoxelNeXtHeadKPMerge': VoxelNeXtHeadKPMerge,
    'TransFusionHead': TransFusionHead,
}
