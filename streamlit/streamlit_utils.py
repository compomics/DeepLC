"""Streamlit utils."""

import base64
import io
import logging
import os
import re
import tempfile
import uuid
import zipfile
from typing import BinaryIO

from typing_extensions import get_origin

import streamlit as st


class StreamlitLogger:
    """Pickup logger and write to Streamlit front end."""

    def __init__(self, placeholder, logger_name=None, accumulate=True, persist=True):
        """
        Pickup logger and write to Streamlit front end.

        Parameters
        ----------
            placeholder: streamlit.empty
                Streamlit placeholder object on which to write logs.
            logger_name: str, optional
                Module name of logger to pick up. Leave to None to pick up root logger.
            accumulate: boolean, optional
                Whether to accumulate log messages or to overwrite with each new
                message, keeping only the last line. (default: True)
            persist: boolean, optional
                Wheter to keep the log when finished, or empty the placeholder element.

        """
        self.placeholder = placeholder
        self.persist = persist

        self.logging_stream = _StreamlitLoggingStream(placeholder, accumulate)
        self.handler = logging.StreamHandler(self.logging_stream)
        self.logger = logging.getLogger(logger_name)

    def __enter__(self):
        self.logger.addHandler(self.handler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self.handler)
        if not self.persist:
            self.placeholder.empty()


class _StreamlitLoggingStream:
    """Logging stream that writes logs to Streamlit front end."""

    def __init__(self, placeholder, accumulate=True):
        """
        Logging stream that writes logs to Streamlit front end.

        Parameters
        ----------
            placeholder: streamlit.empty
                Streamlit placeholder object on which to write logs
            accumulate: boolean, optional
                Whether to accumulate log messages or to overwrite with each new
                message, keeping only the last line (default: True)
        """
        self.placeholder = placeholder
        self.accumulate = accumulate
        self.message_list = []

    def write(self, message):
        if self.accumulate:
            self.message_list.append(message)
        else:
            self.message_list = [message]
        self.placeholder.markdown("```\n" + "".join(self.message_list) + "\n```")


def hide_streamlit_menu():
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def bytesio_to_tempfile(bytesio: BinaryIO) -> str:
    """Write BytesIO object to temporary file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytesio.getbuffer())
        filepath = tmp.name
    return filepath


def zip_files(file_list):
    bytesio = io.BytesIO()
    with zipfile.ZipFile(bytesio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in file_list:
            zf.write(f, arcname=os.path.basename(f))

    bytesio.seek(0)
    return bytesio.read()


def get_zipfile_href(file_list):
    """Generate href for zip download of file list."""
    zipf = zip_files(file_list)
    b64 = base64.b64encode(zipf).decode()
    href = f'data:file/zip;base64,{b64}'
    return href


def encode_object_for_url(object_to_download):
    try:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()
    return b64


def styled_download_button(
    href, button_text, download_filename=None,
):
    """
    Generates a styled button with any given href.

    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220

    Params
    ------
    href
        Link to place in HTML `<a>` href field
    button_text, str
        Text to display on download button (e.g. 'click here to download file')
    download_filename, str (optional)
        If download is a file, add its filename and extension. e.g. mydata.csv,
        some_txt_output.txt

    """
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: {st.config.get_option("theme.backgroundColor")};
                color: {st.config.get_option("theme.textColor")};
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: {st.config.get_option("theme.primaryColor")};
                color: {st.config.get_option("theme.primaryColor")};
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: {st.config.get_option("theme.primaryColor")};
                color: white;
                }}
        </style> """

    dl_string = f'download="{download_filename}"' if download_filename else ''
    dl_link = (
        custom_css
        + f'<a {dl_string} id="{button_id}" href="{href}">{button_text}</a><br><br>'
    )

    st.markdown(dl_link, unsafe_allow_html=True)
