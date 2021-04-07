# DeepLC Streamlit-based web server


## Usage

Pull and run with

```sh
docker run -p 8501 ghcr.io/compomics/deeplc-streamlit
```

Streamlit can be further configured using environment variables:

```sh
docker run \
    -p 8501 \
    -e STREAMLIT_SERVER_MAXUPLOADSIZE="200" \
    -e STREAMLIT_SERVER_BASEURLPATH="deeplc" \
    ghcr.io/compomics/deeplc-streamlit
```
See
[Streamlit configuration](https://docs.streamlit.io/en/stable/streamlit_configuration.html)
for more info.
