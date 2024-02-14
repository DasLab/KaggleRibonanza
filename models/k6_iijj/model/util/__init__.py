__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    raise Exception("version not found")

from .util import *
from .batch_reader import BatchReader
from .basic_models import BasicCFG, CFG, Model, InputFeature
from .text_process import Vocabulary
