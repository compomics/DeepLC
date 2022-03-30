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
        usage="deeplc [<options>] <peptide_file>",
        formatter_class=lambda prog: HelpFormatter(prog, max_help_position=42),
        add_help=False,
    )

    io_args = parser.add_argument_group(
        "Input and output files",
        gooey_options={'columns':2}
    )
    io_args.add_argument(
        "file_pred",
        type=str,
        widget="FileChooser",
        metavar="Input peptides for prediction (required)" if gui else "",
        help="path to peptide CSV file for which to make predictions"
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
    model_cal_args.add_argument(
        "--pygam_calibration",  # TODO: Change into "Do not use ..." with store_false?
        dest="pygam_calibration",
        action='store_true',
        default=False,
        widget="BlockCheckbox",
        metavar="Use pyGAM calibration" if gui else "",
        gooey_options={"checkbox_label": "Enable"},
        help=(
            "calibrate with pyGAM generalized additive model instead of "
            "simple piecewise linear fit"
        )
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
        "--use_library",  # TODO: Change into "Do not use library" with store_false?
        dest="use_library",
        action='store_true',
        default=False,
        widget="BlockCheckbox",
        metavar="Use prediction library" if gui else "",
        gooey_options={"checkbox_label": "Enable"},
        help="use library with previous predictions for faster results",
        # TODO Help says "takes argument with string", but is bool?
    )
    advanced_args.add_argument(
        "--write_library",  # TODO: Change into "Do not use library" with store_false?
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
        default=cpu_count(),
        widget="Slider",
        metavar="Parallelization " if gui else "",
        gooey_options={"min": 1, "max": cpu_count(), "increment": 1},
        help="number of CPU threads to use; default=all"
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

