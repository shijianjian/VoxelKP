from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .pvrcnn_head_kp import PVRCNNHeadKP
from .pvrcnn_head_kp_v2 import PVRCNNHeadKPV2
from .pvrcnn_head_kp_multihead import PVRCNNHeadKPMultihead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate
from .mppnet_head import MPPNetHead
from .mppnet_memory_bank_e2e import MPPNetHeadE2E

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PVRCNNHeadKP': PVRCNNHeadKP,
    'PVRCNNHeadKPV2': PVRCNNHeadKPV2,
    'PVRCNNHeadKPMultihead': PVRCNNHeadKPMultihead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'MPPNetHead': MPPNetHead,
    'MPPNetHeadE2E': MPPNetHeadE2E,
}
