FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install wheel setuptools \
    && pip install "mlflow[extras]==2.22.0"

ENTRYPOINT ["mlflow"]