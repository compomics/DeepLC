"""Graphical user interface."""
import sys
from multiprocessing import freeze_support
from pathlib import Path
import importlib.resources
try:
    from gooey import Gooey, local_resource_path
except ImportError:
    raise ImportError(
        "Missing dependency `gooey` required to start graphical user interface. "
        "Install `gooey` or use the command line interface."
    )

import deeplc.package_data.gui_images as img_module
from deeplc.__main__ import main


# Get path to package_data/images
# Workaround with parent of specific file required for Python 3.9+ support
with importlib.resources.path(img_module, 'config_icon.png') as resource:
    _IMG_DIR = Path(resource).parent


@Gooey(
    program_name="DeepLC",
    image_dir=local_resource_path(_IMG_DIR),
    tabbed_groups=True,
    default_size=(720, 480),
    monospace_display=True,
    target=None if getattr(sys, 'frozen', False) else "deeplc-gui"
)
def start_gui():
    """Run main with GUI enabled."""
    freeze_support()  # Required for multiprocessing with PyInstaller
    main(gui=True)

if __name__ == "__main__":
    start_gui()
