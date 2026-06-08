import torch
import torch.nn as nn

class MusicCRNN(nn.Module):
    def __init__(self, num_classes=12):
        super(MusicCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.rnn = nn.GRU(input_size=256 * 8, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        batch_size, seq_len, channels, height = x.size()
        x = x.reshape(batch_size, seq_len, channels * height)
        out, _ = self.rnn(x)
        out = torch.mean(out, dim=1) # GAP a lo largo del tiempo
        return self.fc(out)