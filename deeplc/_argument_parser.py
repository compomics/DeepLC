import os
from argparse import HelpFormatter
from multiprocessing import cpu_count

try:
    from gooey import GooeyParser as ArgumentParser
except ImportError:
    from argparse import ArgumentParser
    HAS_GOOEY = False
else:
    HAS_GOOEY = True

from deeplc import __version__


def parse_arguments(gui=False):
    """Read arguments from the CLI or GUI."""
    if gui and not HAS_GOOEY:
        raise ImportError(
            "Missing dependency `gooey` required to start graphical user interface. "
            "Install `gooey` or use the command line interface."
        )

    # Define Gooey-specific arguments upfront to be enabled only if Gooey is available
    gooey_args = {
        "io_args": {"gooey_options": {'columns':2}},
        "file_pred": {
            "widget": "FileChooser",
            "metavar": "Input peptides for prediction (required)"
        },
        "file_cal": {"widget": "FileChooser"},
        "file_pred_out": {"widget": "FileSaver"},
        "plot_predictions": {
            "widget": "BlockCheckbox",
            "gooey_options": {"checkbox_label": "Enable"},
            "metavar": "Plot predictions",
        },
        "model_cal_args": {"gooey_options": {'columns':2}},
        "file_model": {"widget": "MultiFileChooser"},
        "calibration_group": {
            "gooey_options": {
                "initial_selection": 0,
                "title": "Calibration method",
                "full_width": True,
            }
        },
        "pygam_calibration": {
            "gooey_options": {"checkbox_label": "Use pyGAM calibration"},
            "metavar": "Use pyGAM calibration"
        },
        "legacy_calibration": {
            "gooey_options": {"checkbox_label": "Use legacy calibration"},
            "metavar": "Use legacy calibration"
        },
        "split_cal": {"gooey_options": {"visible": False}},
        "dict_divider": {"gooey_options": {"visible": False}},
        "advanced_args": {"gooey_options": {'columns':2}},
        "use_library": {"widget": "FileChooser"},
        "write_library": {
            "widget": "BlockCheckbox",
            "gooey_options": {"checkbox_label": "Enable"},
            "metavar": "Append to prediction library",
        },
        "batch_num": {
            "widget": "IntegerField",
            "gooey_options": {"min": 10e3, "max": 10e9, "increment": 1000},
        },
        "n_threads": {
            "widget": "Slider",
            "gooey_options": {"min": 1, "max": cpu_count(), "increment": 1},
        },
        "log_level": {"widget": "Dropdown"},
        "version": {"gooey_options": {"visible": False}},
        "help": {"gooey_options": {"visible": False}},
    }
    if not gui:
        gooey_args = {k: {} for k, v in gooey_args.items()}

    parser = ArgumentParser(
        prog="DeepLC",
        description=(
            "Retention time prediction for (modified) peptides using deep "
            "learning."),
        usage="deeplc [OPTIONS] --file_pred <peptide_file>",
        formatter_class=lambda prog: HelpFormatter(prog, max_help_position=42),
        add_help=False,
    )

    io_args = parser.add_argument_group(
        "Input and output files",
        **gooey_args["io_args"]
    )
    io_args.add_argument(
        "--file_pred",
        required=True,
        type=str,
        help="path to peptide CSV file for which to make predictions (required)",
        **gooey_args["file_pred"]
    )
    io_args.add_argument(
        "--file_cal",
        type=str,
        default=None,
        metavar="Input peptides for calibration" if gui else "",
        help=(
            "path to peptide CSV file with retention times to use for "
            "calibration"
        ),
        **gooey_args["file_cal"]
    )
    io_args.add_argument(
        "--file_pred_out",
        type=str,
        default=None,
        metavar="Output filename" if gui else "",
        help="path to write output file with predictions",
        **gooey_args["file_pred_out"]
    )
    io_args.add_argument(
        "--plot_predictions",
        action='store_true',
        default=False,
        help="save scatter plot of predictions vs observations",
        **gooey_args["plot_predictions"]
    )

    model_cal_args = parser.add_argument_group(
        "Model and calibration",
        **gooey_args["model_cal_args"]
    )
    model_cal_args.add_argument(
        "--file_model",
        nargs="+",
        default=None,
        metavar="Model file(s)" if gui else "",
        help=(
            "path to prediction model(s); leave empty to select the best of "
            "the default models based on the calibration peptides"
        ),
        **gooey_args["file_model"],
    )

    calibration_group = model_cal_args.add_mutually_exclusive_group(
        **gooey_args["calibration_group"]
    )
    calibration_group.add_argument(
        "--pygam_calibration",
        dest="pygam_calibration",
        action="store_true",
        help=(
            "use pyGAM generalized additive model as calibration method; "
            "recommended; default"
        ),
        **gooey_args["pygam_calibration"]
    )
    calibration_group.add_argument(
        "--legacy_calibration",
        dest="pygam_calibration",
        action="store_false",
        help="use legacy simple piecewise linear fit as calibration method",
        **gooey_args["legacy_calibration"]
    )

    model_cal_args.add_argument(
        "--split_cal",
        type=int,
        dest="split_cal",
        default=50,
        metavar="split cal" if gui else "",
        help=(
            "number of splits in the chromatogram for piecewise linear "
            "calibration fit"
        ),
        **gooey_args["split_cal"]
    )
    model_cal_args.add_argument(
        "--dict_divider",
        type=int,
        dest="dict_divider",
        default=50,
        metavar="dict divider" if gui else "",
        help=(
            "sets precision for fast-lookup of retention times for "
            "calibration; e.g. 10 means a precision of 0.1 between the "
            "calibration anchor points"
        ),
        **gooey_args["dict_divider"]
    )

    advanced_args = parser.add_argument_group(
        "Advanced configuration", **gooey_args["advanced_args"]
    )
    advanced_args.add_argument(
        "--use_library",
        dest="use_library",
        action='store',
        default=False,
        metavar="Select prediction library file" if gui else "",
        help=(
            "library file with previous predictions for faster results to "
            "read from, or to write to"
        ),
        **gooey_args["use_library"]
    )
    advanced_args.add_argument(
        "--write_library",
        dest="write_library",
        action='store_true',
        default=False,
        help="append new predictions to library for faster future results",
        **gooey_args["write_library"]
    )
    advanced_args.add_argument(
        "--batch_num",
        type=int,
        dest="batch_num",
        default=250000,
        metavar="Batch size" if gui else "",
        help=(
            "prediction batch size (in peptides); lower to decrease memory "
            "footprint; default=250000"
        ),
        **gooey_args["batch_num"]
    )
    advanced_args.add_argument(
        "--n_threads",
        type=int,
        default=max(cpu_count(), 16),
        metavar="Parallelization " if gui else "",
        help="number of CPU threads to use; default=all with max of 16",
        **gooey_args["n_threads"]
    )
    advanced_args.add_argument(
        "--log_level",
        type=str,
        action="store",
        dest="log_level",
        default='info',
        metavar="Logging level" if gui else "",
        choices=["debug","info","warning","error","critical"],
        help="verbosity of logging messages; default=info",
        **gooey_args["log_level"]
    )
    advanced_args.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
        **gooey_args["version"]
    )
    advanced_args.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
        **gooey_args["help"]
    )

    results = parser.parse_args()

    if not results.file_pred_out:
        results.file_pred_out = os.path.splitext(results.file_pred)[0] + '_deeplc_predictions.csv'

    return results
