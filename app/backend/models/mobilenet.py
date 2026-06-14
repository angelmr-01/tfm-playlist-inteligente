import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetTL(nn.Module):
    """
    MobileNetV3 (Small) adaptado para espectrogramas de 1 canal.
    Utiliza Transfer Learning (pesos por defecto) en la base convolucional,
    pero cambia la primera capa para aceptar 1 canal y reemplaza la cabeza
    de clasificación por un cuello de botella y un promediado temporal.
    """
    def __init__(self, num_classes=12):
        super(MobileNetTL, self).__init__()

        # --- FASE 1: VISIÓN ARTIFICIAL LIGERA (MobileNetV3) ---
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Adaptar primera capa para 1 canal (espectrogramas en escala de grises / dB)
        original_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(1, original_conv.out_channels,
                                             kernel_size=original_conv.kernel_size,
                                             stride=original_conv.stride,
                                             padding=original_conv.padding,
                                             bias=False)

        self.feature_extractor = mobilenet.features

        # --- FASE 1.5: EL CUELLO DE BOTELLA Y EL RESHAPE ---
        self.freq_pool = nn.AdaptiveAvgPool2d((4, None))

        d_model_cnn_out = 576 * 4
        self.d_model = 512

        self.bottleneck = nn.Linear(d_model_cnn_out, self.d_model)

        # --- FASE 4: CLASIFICADOR ---
        self.fc = nn.Linear(self.d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.bottleneck.weight)
        if self.bottleneck.bias is not None:
            nn.init.zeros_(self.bottleneck.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.freq_pool(x)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time, channels * freq)
        x = self.bottleneck(x)
        x = torch.relu(x)
        
        # Colapso del eje temporal (promedio)
        x = torch.mean(x, dim=1)
        
        return self.fc(x)
