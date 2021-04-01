"""Streamlit-based web interface for DeepLC."""

import base64
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from deeplc import DeepLC

from streamlit_utils import StreamlitLogger, hide_streamlit_menu


class DeepLCStreamlitError(Exception):
    pass


class MissingPeptideCSV(DeepLCStreamlitError):
    pass


class MissingCalibrationPeptideCSV(DeepLCStreamlitError):
    pass


class MissingCalibrationColumn(DeepLCStreamlitError):
    pass


class StreamlitUI:
    """DeepLC Streamlit UI."""

    def __init__(self):
        """DeepLC Streamlit UI."""
        self.texts = WebpageTexts
        self.user_input = dict()

        st.set_page_config(
            page_title="DeepLC web server",
            page_icon=":rocket:",
            layout="centered",
            initial_sidebar_state="expanded",
        )

        hide_streamlit_menu()

        self._main_page()
        self._sidebar()

    def _main_page(self):
        """Format main page."""
        st.title("DeepLC")
        st.header("Input and configuration")
        st.subheader("Input files")
        self.user_input["input_csv"] = st.file_uploader(
            "Peptide CSV", help=self.texts.Help.peptide_csv
        )
        self.user_input["input_csv_calibration"] = st.file_uploader(
            "Calibration peptide CSV (optional)",
            help=self.texts.Help.calibration_peptide_csv,
        )
        self.user_input["use_example"] = st.checkbox(
            "Use example data", help=self.texts.Help.example_data
        )

        with st.beta_expander("Info about peptide CSV formatting"):
            st.markdown(self.texts.Help.csv_formatting)

        st.subheader("Calibration")
        self.user_input["calibration_source"] = st.radio(
            "Calibration peptides",
            [
                "Use `tr` column in peptide CSV",
                "Use calibration peptide CSV",
                "Do not calibrate predictions",
            ],
            help=self.texts.Help.calibration_source,
        )
        with st.beta_expander("Set advanced calibration options"):
            self.user_input["dict_cal_divider"] = st.number_input(
                "Dictionary divider",
                step=1,
                value=100,
                help=self.texts.Help.dict_cal_divider,
            )
            self.user_input["split_cal"] = st.number_input(
                "Split calibration", step=1, value=25, help=self.texts.Help.split_cal
            )

        st.subheader("Prediction speed boost")
        self.user_input["use_library"] = st.checkbox(
            "Use prediction library for speed-up", help=self.texts.Help.use_library
        )
        st.markdown(self.texts.Help.use_library_agreement)

        if st.button("Predict retention times"):
            try:
                self._run_deeplc()
            except MissingPeptideCSV:
                st.error(self.texts.Errors.missing_peptide_csv)
            except MissingCalibrationPeptideCSV:
                st.error(self.texts.Errors.missing_calibration_peptide_csv)
            except MissingCalibrationColumn:
                st.error(self.texts.Errors.missing_calibration_column)

    def _sidebar(self):
        """Format sidebar."""
        st.sidebar.image(
            "https://github.com/compomics/deeplc/raw/master/img/deeplc_logo.png",
            width=150,
        )
        st.sidebar.markdown(self.texts.Sidebar.badges)
        st.sidebar.header("About")
        st.sidebar.markdown(self.texts.Sidebar.about, unsafe_allow_html=True)

    def _run_deeplc(self):
        """Run DeepLC given user input, and show results."""
        # Parse user configconfig
        config = self._parse_user_config(self.user_input)
        use_lib = self.user_input["use_library"]
        calibrate = isinstance(config["input_df_calibration"], pd.DataFrame)

        # Run DeepLC and send logs to front end
        st.header("Running DeepLC")
        status_placeholder = st.empty()
        logger_placeholder = st.empty()
        status_placeholder.info(":hourglass_flowing_sand: Running DeepLC...")
        try:
            with StreamlitLogger(logger_placeholder):
                dlc = DeepLC(
                    dict_cal_divider=self.user_input["dict_cal_divider"],
                    split_cal=self.user_input["split_cal"],
                    use_library="deeplc_library.txt" if use_lib else "",
                    write_library=True if use_lib else False,
                    reload_library=True if use_lib else False,
                )
                if calibrate:
                    dlc.calibrate_preds(seq_df=config["input_df_calibration"])
                preds = dlc.make_preds(seq_df=config["input_df"], calibrate=calibrate)
        except Exception as e:
            status_placeholder.error(":x: DeepLC ran into a problem")
            st.exception(e)
        else:
            status_placeholder.success(":heavy_check_mark: Finished!")

            # Add predictions to input DataFrame
            result_df = config["input_df"]
            result_df["predicted_tr"] = preds

            # Show head of result DataFrame
            st.header("Results")
            st.subheader("Selection of predicted retention times")
            st.dataframe(result_df.head(100))

            # Plot results
            self._plot_results(result_df)

            # Download link
            st.subheader("Download predictions")
            filename = os.path.splitext(config["input_filename"])[0]
            self._df_download_href(result_df, filename + "_deeplc_predictions.csv")

    @staticmethod
    def get_example_input():
        """Return example DataFrame for input."""
        return pd.DataFrame(
            [
                ["AAGPSLSHTSGGTQSK", "", 12.1645],
                ["AAINQKLIETGER", "6|Acetyl", 34.095],
                ["AANDAGYFNDEMAPIEVKTK", "12|Oxidation|18|Acetyl", 37.3765],
            ],
            columns=["seq", "modifications", "tr"],
        )

    def _parse_user_config(self, user_input):
        """Validate and parse user input."""
        config = {
            "input_filename": None,
            "input_df": None,
            "input_df_calibration": None,
        }

        # Get peptide dataframe
        if user_input["use_example"]:
            config["input_filename"] = "example.csv"
            config["input_df"] = self.get_example_input()
        elif user_input["input_csv"]:
            config["input_filename"] = user_input["input_csv"].name
            config["input_df"] = pd.read_csv(user_input["input_csv"])
            config["input_df"]["modifications"].fillna("", inplace=True)
        else:
            raise MissingPeptideCSV

        # Get calibration peptide dataframe
        if user_input["calibration_source"] == "Use `tr` column in peptide CSV":
            if "tr" not in config["input_df"].columns:
                raise MissingCalibrationColumn
            else:
                config["input_df_calibration"] = config["input_df"]
        elif user_input["calibration_source"] == "Use calibration peptide CSV":
            if not user_input["input_csv_calibration"]:
                raise MissingCalibrationPeptideCSV
            else:
                config["input_df_calibration"] = pd.read_csv(
                    user_input["input_csv_calibration"]
                )
                config["input_df_calibration"]["modifications"].fillna("", inplace=True)

        return config

    @staticmethod
    def _plot_results(result_df):
        """Plot results with Plotly Express."""
        if "tr" in result_df.columns:
            st.subheader("Input retention times vs predictions")
            fig = px.scatter(
                result_df,
                x="tr",
                y="predicted_tr",
                hover_data=["seq", "modifications"],
                trendline="ols",
                opacity=0.25,
                color_discrete_sequence=["#763737"],
            )
            fig.update_layout(
                xaxis_title_text="Input retention time",
                yaxis_title_text="Predicted retention time",
            )
        else:
            st.subheader("Predicted retention time distribution")
            fig = px.histogram(
                result_df,
                x="predicted_tr",
                marginal="rug",
                opacity=0.8,
                histnorm="density",
                color_discrete_sequence=["#763737"],
            )
            fig.update_layout(
                xaxis_title_text="Predicted retention time",
                yaxis_title_text="Density",
                bargap=0.2,
            )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _df_download_href(df, filename="deeplc_predictions.csv"):
        """Get download href for pd.DataFrame CSV."""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)


class WebpageTexts:
    class Sidebar:
        badges = """
            [![GitHub release](https://img.shields.io/github/release-pre/compomics/deeplc.svg?style=flat-square)](https://github.com/compomics/deeplc/releases)
            [![GitHub](https://img.shields.io/github/license/compomics/deeplc.svg?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)
            [![Twitter](https://flat.badgen.net/twitter/follow/compomics?icon=twitter)](https://twitter.com/compomics)
            """

        about = """
            DeepLC is a retention time predictor for (modified) peptides that employs
            Deep Learning. Its strength lies in the fact that it can accurately predict
            retention times for modified peptides, even if hasn't seen said modification
            during training.

            DeepLC can be run with a
            [graphical user interface](https://github.com/compomics/DeepLC#graphical-user-interface),
            as a[Python package](https://github.com/compomics/DeepLC#python-package)
            (both CLI and Python API), or through this web application.

            If you use DeepLC for your research, please use the following citation:
            >**DeepLC can predict retention times for peptides that carry as-yet unseen
            modifications**<br>
            >Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens, Sven
            Degroeve<br>
            >_bioRxiv (2020)_<br>
            >[doi:10.1101/2020.03.28.013003](https://doi.org/10.1101/2020.03.28.013003)
            """

    class Help:
        peptide_csv = """
            CSV with peptides for which to predict retention times. Click below on _Info
            about peptide CSV formatting_ for more info.
            """
        calibration_peptide_csv = """
            CSV with peptides with known retention times to be used for calibration.
            Click below on _Info about peptide CSV formatting_ for more info.
            """
        example_data = "Use example data instead of uploaded CSV files."
        csv_formatting = """
            DeepLC expects comma-separated values (CSV) with the following columns:

            - `seq`: Unmodified peptide sequences
            - `modifications`: MS²PIP-style formatted peptide modifications: Each
            modification is listed as `location|name`, separated by a pipe (`|`) between
            the location, the name, and other modifications. `location` is an integer
            counted starting at 1 for the first AA. `0` is reserved for N-terminal
            modifications, `-1` for C-terminal modifications. `name` has to correspond
            to a Unimod (PSI-MS) name. All supported modifications are listed on
            [GitHub](https://github.com/compomics/DeepLC/blob/master/deeplc/unimod/unimod_to_formula.csv)
            - `tr`: Retention time (only required for calibration CSV)

            For example:

            ```csv
            seq,modifications,tr
            AAGPSLSHTSGGTQSK,,12.1645
            AAINQKLIETGER,6|Acetyl,34.095
            AANDAGYFNDEMAPIEVKTK,12|Oxidation|18|Acetyl,37.3765
            ```

            See
            [examples/datasets](https://github.com/compomics/DeepLC/tree/master/examples/datasets)
            for more examples.
            """
        calibration_source = """
            DeepLC can calibrate its predictions based on set of known peptide retention
            times. Calibration also ensures that the best-fitting DeepLC model is used.
            """
        dict_cal_divider = """
            This parameter defines the precision to use for fast-lookup of retention
            times for calibration. A value of 10 means a precision of 0.1 (and 100 a
            precision of 0.01) between the calibration anchor points. This parameter
            does not influence the precision of the calibration, but setting it too
            high results in mean that there is bad selection of the models between
            anchor points. A safe value is usually higher than 10.
            """
        split_cal = """
            The number of splits for the chromatogram. If the value is set to 10 the
            chromatogram is split up into 10 equidistant parts. For each part the median
            value of the calibration peptides is selected. These are the anchor points.
            Between each anchor point a linear model is fit.
            """
        use_library = """
            DeepLC can fetch previously predicted retention times from a library,
            instead predicting retention times for the same (modified) peptide again.
            This feature will not change any of the predicted retention time values. It
            can, however, significantly speed up DeepLC.
            """
        use_library_agreement = """
            _By selecting this box, you allow us to store the uploaded peptide sequences
            and modifications on this server indefinitely._
            """

    class Errors:
        missing_peptide_csv = """
            Upload a peptide CSV file or select the _Use example data_ checkbox.
            """
        missing_calibration_peptide_csv = """
            Upload a calibration peptide CSV file or select another _Calibration
            peptides_ option.
            """
        missing_calibration_column = """
            Upload a peptide CSV file with a `tr` column or select another _Calibration
            peptides_ option.
            """


if __name__ == "__main__":
    StreamlitUI()
