FROM python:3.10

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Fix: uploads folder pehle banao aur permission do
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]