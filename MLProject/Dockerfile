FROM python:3.10-slim

WORKDIR /app

COPY . .

# Upgrade pip dan install mlflow serta perbaiki dependency packaging
RUN pip install --upgrade pip setuptools wheel \
    && pip install "mlflow[extras]==2.2.2" \
    && pip install packaging==23.2  # ✅ Fix error import Version

ENTRYPOINT ["mlflow"]
