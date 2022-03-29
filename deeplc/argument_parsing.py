

def parse_arguments():
    """Read arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_pred",
        type=str,
        dest="file_pred",
        default="",
        help="Path to peptide file for which to make predictions (required)")

    parser.add_argument(
        "--file_cal",
        type=str,
        dest="file_cal",
        default="",
        help="Path to peptide file with retention times to use for calibration\
            (optional)")

    parser.add_argument(
        "--file_pred_out",
        type=str,
        dest="file_pred_out",
        default="",
        help="Path to output file with predictions (optional)")

    parser.add_argument(
        "--file_model",
        help="Path to prediction model(s). Seperate with spaces. Leave empty \
            to select the best of the default models (optional)",
        nargs="+",
        default=None
    )

    parser.add_argument(
        "--split_cal",
        type=int,
        dest="split_cal",
        default=50,
        # TODO add help
        )

    parser.add_argument(
        "--dict_divider",
        type=int,
        dest="dict_divider",
        default=50,
        # TODO add help
        )

    parser.add_argument(
        "--batch_num",
        type=int,
        dest="batch_num",
        default=250000,
        help="Batch size (in peptides) for predicting the retention time. Set\
            lower to decrease memory footprint (optional, default=250000)")

    parser.add_argument(
        "--plot_predictions",
        dest='plot_predictions',
        action='store_true',
        default=False,
        help='Save scatter plot of predictions vs observations (default=False)'
    )

    parser.add_argument(
        "--n_threads",
        type=int,
        dest="n_threads",
        default=16,
        help="Number of threads to use (optional, default=maximum available)")

    parser.add_argument(
        "--log_level",
        type=str,
        dest="log_level",
        default='info',
        help="Logging level (debug, info, warning, error, or critical; default=info)"
    )

    parser.add_argument(
        "--use_library",
        type=str,
        dest="use_library",
        default="",
        help="Use a library with previously predicted retention times, argument takes a string with the location to the library"
    )

    parser.add_argument(
        "--write_library",
        dest="write_library",
        action='store_true',
        default=False,
        help="Append to a library with predicted retention times, will write to the file specified by --use_library"
    )

    parser.add_argument(
        "--pygam_calibration",
        dest="pygam_calibration",
        action='store_true',
        default=False,
        help="Append to a library with predicted retention times, will write to the file specified by --use_library"
    )

    parser.add_argument("--version", action="version", version=__version__)

    results = parser.parse_args()

    if not results.file_pred:
        parser.print_help()
        exit(0)

    if not results.file_pred_out:
        results.file_pred_out = os.path.splitext(results.file_pred)[0] + '_deeplc_predictions.csv'

    return results


def parse_arguments_gooey():
    parser = GooeyParser(description="DeepLC GUI")

    parser.add_argument('Filename', widget="FileChooser")
    parser.add_argument('Date', widget="DateChooser")

    parser.add_argument(
        "--file_pred",
        type=str,
        dest="file_pred",
        default="",
        widget="FileChooser",
        help="Path to peptide file for which to make predictions (required)")

    parser.add_argument(
        "--file_cal",
        type=str,
        dest="file_cal",
        default="",
        widget="FileChooser",
        help="Path to peptide file with retention times to use for calibration\
            (optional)")

    parser.add_argument(
        "--file_pred_out",
        type=str,
        dest="file_pred_out",
        default="",
        widget="FileChooser",
        help="Path to output file with predictions (optional)")

    parser.add_argument(
        "--file_model",
        help="Path to prediction model(s). Seperate with spaces. Leave empty \
            to select the best of the default models (optional)",
        nargs="+",
        widget="DirChooser",
        default=None
    )

    parser.add_argument(
        "--batch_num",
        type=int,
        dest="batch_num",
        default=250000,
        widget="IntegerField",
        help="Batch size (in peptides) for predicting the retention time. Set\
            lower to decrease memory footprint (optional, default=250000)")

    parser.add_argument(
        "--plot_predictions",
        dest='plot_predictions',
        action='store_true',
        default=False,
        widget="CheckBox",
        help='Save scatter plot of predictions vs observations (default=False)'
    )

    parser.add_argument(
        "--n_threads",
        type=int,
        dest="n_threads",
        default=16,
        widget="IntegerField",
        help="Number of threads to use (optional, default=maximum available)")

    parser.add_argument(
        "--log_level",
        type=str,
        action="store",
        dest="log_level",
        default='info',
        widget="Dropdown",
        choices=["debug","info","warning","error","critical"],
        help="Logging level (debug, info, warning, error, or critical; default=info)"
    )

    parser.add_argument(
        "--use_library",
        type=bool,
        dest="use_library",
        default=False,
        widget="CheckBox",
        help="Use a library with previously predicted retention times, argument takes a string with the location to the library"
    )

    parser.add_argument(
        "--write_library",
        dest="write_library",
        action='store_true',
        default=False,
        widget="CheckBox",
        help="Append to a library with predicted retention times, will write to the file specified by --use_library"
    )

    parser.add_argument(
        "--pygam_calibration",
        dest="pygam_calibration",
        action='store_true',
        default=False,
        widget="CheckBox",
        help="Append to a library with predicted retention times, will write to the file specified by --use_library"
    )

    results = parser.parse_args()

    if not results.file_pred_out:
        results.file_pred_out = os.path.splitext(results.file_pred)[0] + '_deeplc_predictions.csv'

    return results

