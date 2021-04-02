import pkg_resources

from deeplc.deeplc import DeepLC
from deeplc.feat_extractor import FeatExtractor


__version__ = pkg_resources.require("deeplc")[0].version
