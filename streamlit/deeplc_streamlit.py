"""Streamlit-based web interface for DeepLC."""

import base64
import logging
import os

import pandas as pd
import plotly.express as px

import streamlit as st
from deeplc import DeepLC
from streamlit_utils import StreamlitLogger, hide_streamlit_menu


class StreamlitUI:
    """DeepLC Streamlit UI."""

    def __init__(self):
        """DeepLC Streamlit UI."""
        self.user_input = dict()

        st.set_page_config(
            page_title="DeepLC webserver",
            page_icon=":rocket:",
            layout="centered",
            initial_sidebar_state="expanded",
        )

        # hide_streamlit_menu()  # TODO: Hide menu or not?

        # self._sidebar() # TODO: Add sidebar?
        self._main_page()

    def _main_page(self):
        st.title("DeepLC")
        st.header("About")
        # TODO: Add 'About' text
        st.markdown(
            """
            Lorem ipsum...
            """
        )

        st.header("Input")
        self.user_input["input_csv"] = st.file_uploader("Input peptide CSV")
        self.user_input["use_example"] = st.checkbox("Use example input")

        if st.button("Predict retention times"):
            self._run_deeplc()

    def _sidebar(self):
        pass

    def _run_deeplc(self):
        # Get config
        config = self._parse_user_config(self.user_input)
        if not config:
            return None

        # Run DeepLC and send logs to front end
        st.header("Running DeepLC")
        logger_placeholder = st.empty()
        with StreamlitLogger(logger_placeholder):
            logging.info("Starting DeepLC...")
            dlc = DeepLC()
            # dlc.calibrate_preds(seq_df=cal_df) # TODO: Implement calibration?
            preds = dlc.make_preds(seq_df=config["input_df"], calibrate=False)
            logging.info(":heavy_check_mark: Finished!")

        # Process results
        result_df = config["input_df"]
        result_df["predicted_tr"] = preds

        # Show head of result df
        st.header("Results")
        st.subheader("Selection of predicted retention times")
        st.dataframe(result_df.sample(5 if len(result_df) > 5 else len(result_df)))

        # Scatterplot
        if "tr" in result_df.columns:
            st.subheader("Input retention times vs predictions")
            self._plot_results(result_df)

        # Download link
        st.subheader("Download predictions")
        filename = os.path.splitext(config["input_filename"])[0]
        self._df_download_href(result_df, filename + "_deeplc_predictions.csv")

    @staticmethod
    def get_example_input():
        return pd.DataFrame(
            [
                ["AAGPSLSHTSGGTQSK", "", 12.1645],
                ["AAINQKLIETGER", "6|Acetyl", 34.095],
                ["AANDAGYFNDEMAPIEVKTK", "12|Oxidation|18|Acetyl", 37.3765],
            ],
            columns=["seq", "modifications", "tr"],
        )

    def _parse_user_config(self, user_input):
        if user_input["use_example"]:
            config = {
                "input_filename": "example.csv",
                "input_df": self.get_example_input(),
            }
        elif user_input["input_csv"]:
            config = {
                "input_filename": user_input["input_csv"].name,
                "input_df": pd.read_csv(user_input["input_csv"]),
            }
            config["input_df"]["modifications"].fillna("", inplace=True)
        else:
            st.error(
                "Please upload an input peptide CSV file or choose to use the example "
                "input."
            )
            return None
        return config

    @staticmethod
    def _plot_results(result_df):
        fig = px.scatter(
            result_df,
            x="tr",
            y="predicted_tr",
            hover_data=["seq", "modifications"],
            trendline="ols",
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _df_download_href(df, filename="deeplc_predictions.csv"):
        """Get download href for pd.DataFrame CSV."""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    StreamlitUI()
