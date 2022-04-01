import os
from argparse import HelpFormatter
from multiprocessing import cpu_count

from gooey import GooeyParser

from deeplc import __version__


def parse_arguments(gui=False):
    """Read arguments from the CLI or GUI."""

    parser = GooeyParser(
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
        gooey_options={'columns':2}
    )
    io_args.add_argument(
        "--file_pred",
        required=True,
        type=str,
        widget="FileChooser",
        metavar="Input peptides for prediction (required)" if gui else "",
        help="path to peptide CSV file for which to make predictions (required)"
    )
    io_args.add_argument(
        "--file_cal",
        type=str,
        default=None,
        widget="FileChooser",
        metavar="Input peptides for calibration" if gui else "",
        help=(
            "path to peptide CSV file with retention times to use for "
            "calibration"
        )
    )
    io_args.add_argument(
        "--file_pred_out",
        type=str,
        default=None,
        widget="FileSaver",
        metavar="Output filename" if gui else "",
        help="path to write output file with predictions"
    )
    io_args.add_argument(
        "--plot_predictions",
        action='store_true',
        default=False,
        widget="BlockCheckbox",
        metavar="Plot predictions" if gui else "",
        gooey_options={"checkbox_label": "Enable"},
        help="save scatter plot of predictions vs observations"
    )

    model_cal_args = parser.add_argument_group(
        "Model and calibration", gooey_options={'columns':2}
    )
    model_cal_args.add_argument(
        "--file_model",
        nargs="+",
        default=None,
        metavar="Model file(s)" if gui else "",
        widget="MultiFileChooser",
        help=(
            "path to prediction model(s); leave empty to select the best of "
            "the default models based on the calibration peptides"
        ),
    )

    calibration_group = model_cal_args.add_mutually_exclusive_group(
        gooey_options={
            "initial_selection": 0,
            "title": "Calibration method",
            "full_width": True,
        }
    )
    calibration_group.add_argument(
        "--pygam_calibration",
        dest="pygam_calibration",
        action="store_true",
        metavar="Use pyGAM calibration" if gui else "",
        gooey_options={"checkbox_label": "Use pyGAM calibration"},
        help=(
            "use pyGAM generalized additive model as calibration method; "
            "recommended; default"
        ),
    )
    calibration_group.add_argument(
        "--legacy_calibration",
        dest="pygam_calibration",
        action="store_false",
        metavar="Use legacy calibration" if gui else "",
        gooey_options={"checkbox_label": "Use legacy calibration"},
        help="use legacy simple piecewise linear fit as calibration method",
    )

    model_cal_args.add_argument(
        "--split_cal",
        type=int,
        dest="split_cal",
        default=50,
        metavar="split cal" if gui else "",
        gooey_options={"visible": False},
        help=(
            "number of splits in the chromatogram for piecewise linear "
            "calibration fit"
        ),
    )
    model_cal_args.add_argument(
        "--dict_divider",
        type=int,
        dest="dict_divider",
        default=50,
        metavar="dict divider" if gui else "",
        gooey_options={"visible": False},
        help=(
            "sets precision for fast-lookup of retention times for "
            "calibration; e.g. 10 means a precision of 0.1 between the "
            "calibration anchor points"
        )
    )

    advanced_args = parser.add_argument_group(
        "Advanced configuration", gooey_options={'columns':2}
    )
    advanced_args.add_argument(
        "--use_library",
        dest="use_library",
        action='store',
        default=False,
        widget="FileChooser",
        metavar="Select prediction library file" if gui else "",
        help=(
            "library file with previous predictions for faster results to "
            "read from, or to write to"
        ),
    )
    advanced_args.add_argument(
        "--write_library",
        dest="write_library",
        action='store_true',
        default=False,
        widget="BlockCheckbox",
        metavar="Append to prediction library" if gui else "",
        gooey_options={"checkbox_label": "Enable"},
        help="append new predictions to library for faster future results"
    )
    advanced_args.add_argument(
        "--batch_num",
        type=int,
        dest="batch_num",
        default=250000,
        widget="IntegerField",
        metavar="Batch size" if gui else "",
        gooey_options={"min": 10e3, "max": 10e9, "increment": 1000},
        help=(
            "prediction batch size (in peptides); lower to decrease memory "
            "footprint; default=250000"
        )
    )
    advanced_args.add_argument(
        "--n_threads",
        type=int,
        default=max(cpu_count(), 16),
        widget="Slider",
        metavar="Parallelization " if gui else "",
        gooey_options={"min": 1, "max": cpu_count(), "increment": 1},
        help="number of CPU threads to use; default=all with max of 16"
    )
    advanced_args.add_argument(
        "--log_level",
        type=str,
        action="store",
        dest="log_level",
        default='info',
        widget="Dropdown",
        metavar="Logging level" if gui else "",
        choices=["debug","info","warning","error","critical"],
        help="verbosity of logging messages; default=info"
    )
    advanced_args.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
        gooey_options={"visible": False},
    )
    advanced_args.add_argument(
        "-h",
        "--help",
        action="help",
        gooey_options={"visible": False},
        help="show this help message and exit"
    )

    results = parser.parse_args()

    if not results.file_pred_out:
        results.file_pred_out = os.path.splitext(results.file_pred)[0] + '_deeplc_predictions.csv'

    return results

