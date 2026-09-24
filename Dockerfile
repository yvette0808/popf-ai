FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY src ./src
COPY data/processed/272_candidate/metadata_raw.csv ./data/processed/272_candidate/metadata_raw.csv
COPY data/processed/modern_dataset ./data/processed/modern_dataset
COPY data/raw/脸谱/整理工作/爬虫脸谱图集.zip ./data/raw/脸谱/整理工作/爬虫脸谱图集.zip
COPY artifacts/models/resnet18_conservative_finetune/best_val_macro_f1.pt ./artifacts/models/resnet18_conservative_finetune/best_val_macro_f1.pt

RUN mkdir -p artifacts/feedback artifacts/logs

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
