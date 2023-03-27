from .layoutlm_v3 import LayoutLMv3ForKIE
from .layoutlm_v2 import LayoutLMv2ForKIE

__mapping__ = {
    'layoutlm_v2': LayoutLMv2ForKIE,
    'layoutlm_v3': LayoutLMv3ForKIE,
}
