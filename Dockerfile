FROM python:3.9
COPY . /app
WORKDIR /app
COPY requirement.txt ./requirement.txt
RUN pip install -r requirement.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["Test.py"]