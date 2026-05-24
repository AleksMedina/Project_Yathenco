FROM python:3.11-slim

# Установка системных библиотек для OpenCV (libgl1 вместо устаревшего libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Запуск Uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]