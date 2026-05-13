FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create uploads directory and set permissions for Hugging Face (user 1000)
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Expose port (Hugging Face default is 7860)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
