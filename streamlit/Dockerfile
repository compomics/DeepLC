FROM python:3.8-buster
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "deeplc_streamlit.py"]
