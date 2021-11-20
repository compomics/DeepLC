import sys


if sys.version_info >= (3,8):
    from importlib.metadata import version
    __version__ = version('deeplc')
else:
    import pkg_resources
    __version__ = pkg_resources.require("deeplc")[0].version


from deeplc.deeplc import DeepLC
from deeplc.feat_extractor import FeatExtractor
