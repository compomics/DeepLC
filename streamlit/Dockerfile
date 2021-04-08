FROM continuumio/miniconda3:4.9.2 AS build
COPY environment.yml /
RUN conda env create -f environment.yml && \
    conda install -c conda-forge conda-pack && \
    conda-pack -n deeplc-streamlit -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

FROM debian:buster-slim AS runtime
WORKDIR /deeplc
COPY --from=build /venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY . /deeplc
RUN pip install "deeplc>=0.1.29"
ENV DEEPLC_LIBRARY_PATH="/tmp/deeplc_library.txt"
EXPOSE 8501
CMD ["streamlit", "run", "deeplc_streamlit.py"]
