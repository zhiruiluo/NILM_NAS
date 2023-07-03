from .shared_bce_linear import SharedBCELinear
from .multilabel_linear import MultilabelLinear
from .multilabel_linear_focal import MultilabelLinearFocal
from .shared_bce_plain_linear import SharedBCEPlainLinear
from .multilabel_linear_maskout import MultilabelLinearMask, MultilabelLinearMaskFocal
from .multilabel_linear_asl import MultilabelLinearASL
from .multilabel_sharelinear_asl import MultilabelSharedLinearASL

__all__ = [
    "SharedBCELinear",
    "MultilabelLinear",
    "MultilabelLinearFocal",
    "SharedBCEPlainLinear",
    "MultilabelLinearMask",
    "MultilabelLinearASL",
    "MultilabelLinearMaskFocal",
    "MultilabelSharedLinearASL",
    "HeadModule",
]

HeadModule = {
    'CE': lambda in_dim, nclass, dropout: MultilabelLinear(in_dim, nclass, dropout),
    'Focal': lambda in_dim, nclass, dropout: MultilabelLinearFocal(in_dim, nclass, dropout),
    'SBCE': lambda in_dim, nclass, dropout: SharedBCELinear(in_dim, nclass, dropout),
    'Mask': lambda in_dim, nclass, dropout: MultilabelLinearMask(in_dim, nclass, dropout),
    'MaskFocal': lambda in_dim, nclass, dropout: MultilabelLinearMaskFocal(in_dim, nclass, dropout),
    'ASL': lambda in_dim, nclass, dropout : MultilabelLinearASL(in_dim, nclass, dropout),
    'SASL': lambda in_dim, nclass, dropout: MultilabelSharedLinearASL(in_dim, nclass, dropout),
}