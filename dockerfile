FROM python:3.8-slim
COPY . /mnt/test/flask
WORKDIR /mnt/test/flask
RUN pip install flask gunicorn numpy sklearn scipy xgboost
