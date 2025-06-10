# MLProject/Dockerfile
# Base image untuk MLflow project
FROM python:3.9-slim-buster

# Mengatur working directory di dalam container
WORKDIR /app

# Menginstal dependensi yang dibutuhkan untuk menjalankan project
# Kita asumsikan Conda environment akan dibangun oleh MLflow Project
# Namun, beberapa dependensi dasar mungkin diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy semua file dari project ke dalam container
# MLflow akan menangani bagian ini ketika build-docker dipanggil
# namun untuk Dockerfile dasar, kita bisa menambahkan ini
COPY . .

# Ini adalah placeholder. MLflow build-docker akan menambahkan entrypoint
# yang menjalankan MLflow Project Anda.
# Contoh: CMD ["mlflow", "run", ".", "--entry-point", "main"]
