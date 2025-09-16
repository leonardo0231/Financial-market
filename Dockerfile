
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# متغیرهای عمومی
ARG APP_VERSION=1.0.1
ENV APP_VERSION=${APP_VERSION} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# وابستگی‌های سیستم مورد نیاز برای نصب چرخک‌ها و ابزارهای کوچک
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl ca-certificates tzdata build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# دایرکتوری اپ
WORKDIR /app

# فایل‌های وابستگی
COPY requirements.txt /app/requirements.txt

# نصب وابستگی‌ها
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# کپی سورس
COPY . /app

# ایجاد دایرکتوری‌های لازم
RUN mkdir -p /app/logs /app/config \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

# Labels
LABEL org.opencontainers.image.title="XAUUSD Trading Bot" \
        org.opencontainers.image.version="${APP_VERSION}" \
        org.opencontainers.image.description="Advanced AI-Powered Trading Bot for XAU/USD (Python-only orchestration)" \
        org.opencontainers.image.source="local" \
        org.opencontainers.image.licenses="Proprietary"

USER appuser

EXPOSE 5000 8000

# اجرای مستقیم اسکریپت اصلی Flask (main.py) – خودِ برنامه Flask را روی 0.0.0.0:5000 بالا می‌آورد
ENTRYPOINT ["python", "-m", "xauusd_trading_bot.main"]
