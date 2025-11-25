from mmwave3d.registry import MODELS

import torch
import torch.nn as nn

@MODELS.register_module()
class ExampleTransformerBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(ExampleTransformerBackbone, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x