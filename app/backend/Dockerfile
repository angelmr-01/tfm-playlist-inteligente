# 1. Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# 2. Instalar dependencias del sistema operativo ( para Librosa y procesar MP3 )
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Establecer la carpeta de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiar los requirements y descargarlos
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# 5. Copiar el resto del código y el modelo (.pth) al contenedor
COPY . .

# 6. Exponer el puerto de FastAPI
EXPOSE 8000

# 7. Comando para arrancar el servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]