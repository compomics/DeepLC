"""Graphical user interface."""
import importlib.resources

from gooey import Gooey, local_resource_path

from deeplc import package_data
from deeplc.__main__ import main


with importlib.resources.path(package_data, "gui_images") as img_dir:
    _IMG_DIR = img_dir


@Gooey(
    program_name="DeepLC",
    image_dir=local_resource_path(_IMG_DIR),
    tabbed_groups=True,
    default_size=(760, 720),
    target="deeplc",
    suppress_gooey_flag=True,
    monospace_display=True,
)
def start_gui():
    """Run main with GUI enabled."""
    main(gui=True)

if __name__ == "__main__":
    start_gui()
