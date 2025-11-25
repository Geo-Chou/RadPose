from mmwave3d.registry import MODELS
import torch
import torch.nn as nn

MODELS.register_module()
class SimpleRAECNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, data, label=None):
        x = data.unsqueeze(1)  # [B,1,R,A,E]
        feat = self.net(x).flatten(1)
        logits = self.fc(feat)

        return dict(logits=logits, label=label)