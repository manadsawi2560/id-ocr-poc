import torch, torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),          # H: 48 -> 24
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),        # H: 24 -> 12
            nn.Conv2d(128,256,3,1,1), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(), nn.MaxPool2d((2,1),(2,1)),# H: 12 -> 6
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1),(2,1)), # H: 6 -> 3
            # ใช้ kernel สูง = 3 เพื่อให้ H: 3 -> 1
            nn.Conv2d(512,512,(3,1),1,0), nn.ReLU()
        )
        self.rnn = nn.LSTM(512,256,2,batch_first=True,bidirectional=True)
        self.fc  = nn.Linear(512, num_classes)

    def forward(self, x):        # x: [B,1,H,W]  (H=48)
        feat = self.cnn(x)       # [B,C,H',W']
        B,C,H,W = feat.size()
        if H != 1:
            feat = nn.functional.adaptive_avg_pool2d(feat, (1, W))
            B,C,H,W = feat.size()  
        seq = feat.squeeze(2).permute(0,2,1)  # [B,W',C]
        seq,_ = self.rnn(seq)                 # [B,W',512]
        logits = self.fc(seq)                 # [B,W',num_classes]
        return logits
