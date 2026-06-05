from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os

# Librerías de tu IA
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

# 1. INICIALIZAR EL SERVIDOR WEB (Como hacer const app = express())
app = FastAPI(title="API de Emociones Musicales TFM")

# Configurar CORS para permitir que React (que estará en otro puerto) se comunique
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se pone la URL exacta de React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DEFINIR LA ARQUITECTURA DE LA IA
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. CARGAR EL MODELO EN MEMORIA (Cold Start)
print("⏳ Arrancando servidor y cargando IA en memoria...")
modelo = SimpleCNN(num_classes=4)

# 🔥 IMPORTANTE: Pon aquí la ruta real de tu archivo .pth local
RUTA_CHECKPOINT = ".../Checkpoint/checkpoint_epoch_5.pth" 

if os.path.exists(RUTA_CHECKPOINT):
    checkpoint = torch.load(RUTA_CHECKPOINT, map_location=torch.device('cpu'), weights_only=False)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()
    print("✅ IA cargada y lista para recibir peticiones web.")
else:
    print(f"⚠️ ADVERTENCIA: No se encontró el modelo en {RUTA_CHECKPOINT}")


# 4. DEFINIR LAS RUTAS (ENDPOINTS)

# Ruta base de comprobación (Health Check)
@app.get("/")
def leer_raiz():
    return {"mensaje": "Servidor IA funcionando correctamente", "estado": "online"}

# Ruta principal: Recibe un archivo, lo analiza y devuelve el JSON
@app.post("/api/analizar")
async def analizar_audio(file: UploadFile = File(...)):
    # a) Guardar el archivo temporalmente en el disco
    ruta_temp = f"temp_{file.filename}"
    with open(ruta_temp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # b) Preprocesamiento (Librosa)
        y, sr = librosa.load(ruta_temp, sr=22050, duration=10)
        muestras_esperadas = 10 * 22050
        if len(y) < muestras_esperadas:
            y = np.pad(y, (0, muestras_esperadas - len(y)))
            
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        
        # c) Inferencia (PyTorch)
        tensor_input = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            salida = modelo(tensor_input)
            probabilidades = F.softmax(salida, dim=1)[0].numpy() * 100
            clase_ganadora = int(np.argmax(probabilidades))
            
        diccionario_emociones = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4'}
        emocion_final = diccionario_emociones[clase_ganadora]
        
        # d) Borrar el archivo temporal para no llenar el disco duro
        os.remove(ruta_temp)
        
        # e) Devolver la respuesta a React en formato JSON puro
        # Convertimos explícitamente a float nativo de Python para que JSON lo entienda
        return {
            "archivo": file.filename,
            "emocion_detectada": emocion_final,
            "confianza": {
                "Q1": float(round(probabilidades[0], 2)),
                "Q2": float(round(probabilidades[1], 2)),
                "Q3": float(round(probabilidades[2], 2)),
                "Q4": float(round(probabilidades[3], 2))
            }
        }
        
    except Exception as e:
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)
        return {"error": str(e)}

# Bloque de arranque si se ejecuta este archivo directamente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)