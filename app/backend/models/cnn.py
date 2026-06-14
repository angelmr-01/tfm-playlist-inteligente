import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()

        # --- FASE 1: OÍDO INTERNO (CNN Nativa) ---
        # Mantenemos exactamente los mismos filtros para una comparativa justa
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )

        d_model_cnn_out = 256 * 16

        # Mantenemos el cuello de botella para no alterar el número de parámetros del extractor
        self.bottleneck = nn.Linear(d_model_cnn_out, 512)

        # --- FASE 2 y 3 ELIMINADAS (Sin Inyección de Posición ni Transformer) ---

        # --- FASE 4: CLASIFICADOR MULTI-ETIQUETA ---
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 1. Extracción local
        x = self.cnn(x)

        # 2. Reshape de los tensores
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time, channels * freq)

        # Reducimos a 512 dimensiones
        x = self.bottleneck(x)
        x = torch.relu(x)

        # 3. Global Average Pooling en el tiempo y Clasificación Directa
        # Al no haber Transformer, colapsamos el tiempo directamente aquí
        x = torch.mean(x, dim=1)
        return self.fc(x)