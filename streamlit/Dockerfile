FROM python:3.10-slim-buster
COPY . /deeplc
WORKDIR /deeplc
RUN pip install --upgrade pip && pip install -r requirements.txt
ENV DEEPLC_LIBRARY_PATH="/tmp/deeplc_library.txt"
EXPOSE 8501
CMD ["streamlit", "run", "deeplc_streamlit.py"]
