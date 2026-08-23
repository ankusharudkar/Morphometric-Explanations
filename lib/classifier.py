import torch
from torch import nn

class Classifier1(nn.Module):
    """
    Simple classifier with no residual layers
    """
    def __init__(self, norm = lambda x: nn.InstanceNorm3d(x), pool = lambda : nn.MaxPool3d(2,2)):
        super().__init__()
        dropout = 0.1
        act = lambda: nn.ReLU()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 128, 11, 1),
            norm(128),
            act(),
            pool(),

            nn.Conv3d(128, 256, 5, 1),
            norm(256),
            act(),
            pool(),
            nn.Conv3d(256, 512, 5, 1),
            norm(512),
            act(),
            pool(),

            nn.Conv3d(512, 512, 3, 1),
            norm(512),
            act(),
            nn.Conv3d(512, 512, 3, 1),
            norm(512),
            act(),
            pool(),
            
            nn.Conv3d(512, 512, 1, 1),
            norm(512),
            act(),
            nn.Dropout(dropout),
            nn.Conv3d(512, 512, 1, 1),
            norm(512),
            act(),
            nn.Dropout(dropout),
            nn.Conv3d(512, 1, 1, 1),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out1 = out.view(x.shape[0], -1)
        out2 = out1.mean(dim=-1) 
        return out2, out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, size, 1, size//2),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, size, 1, size//2),
            nn.InstanceNorm3d(out_channels),
        )

        if in_channels != out_channels:
            self.conv1x1 = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out = self.layers(x)
        if self.conv1x1 is not None:
            return out + self.conv1x1(x)
        else:
            return out + x
        
class ResClassifier1(nn.Module):
    def __init__(self, n_out=1):
        super().__init__()
        dropout = 0.05
        act = lambda: nn.ReLU()
        self.conv_layers = nn.Sequential(
            ResBlock(1, 64, 11),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(64, 128, 5),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(128, 256, 5),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(256, 512, 3),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(512, 512, 3),
            act(),
            nn.Dropout(dropout),
            nn.MaxPool3d(2, 2),
            ResBlock(512, 1024, 1),
            act(),
            nn.Dropout(dropout),
            ResBlock(1024, n_out, 1),
        )
        

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = out.mean(dim=-1) 
        return out

class Classifier1_nomax(nn.Module):
    """
    Simple classifier with no residual layers
    """
    def __init__(self, norm = lambda x: nn.InstanceNorm3d(x)):
        super().__init__()
        dropout = 0.1
        act = lambda: nn.ReLU()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 128, 11, 1),
            norm(128),
            act(),
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 128, 3, 2),
            norm(128),
            act(),

            nn.Conv3d(128, 256, 5, 1),
            norm(256),
            act(),
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(256, 256, 3, 2),
            norm(256),
            act(),

            nn.Conv3d(256, 512, 5, 1),
            norm(512),
            
            act(),
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 512, 3, 2),
            norm(512),
            act(),

            nn.Conv3d(512, 512, 3, 1),
            norm(512),
            act(),

            nn.Conv3d(512, 512, 3, 1),
            norm(512),
            act(),
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 512, 3, 2),
            norm(512),
            act(),

            nn.Conv3d(512, 512, 1, 1),
            norm(512),
            act(),

            nn.Dropout(dropout),
            nn.Conv3d(512, 512, 1, 1),
            norm(512),

            act(),
            nn.Dropout(dropout),
            nn.Conv3d(512, 1, 1, 1),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(x.shape[0], -1)
        out = out.mean(dim=-1) 
        return out
    
        
class ResClassifier1_nomax(nn.Module):
    def __init__(self, n_out=1):
        super().__init__()
        dropout = 0.05
        act = lambda: nn.ReLU()
        self.conv_layers = nn.Sequential(
            ResBlock(1, 64, 11),
            act(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 64, 3, 2),
            nn.BatchNorm3d(64),
            act(),
            ResBlock(64, 128, 5),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(128, 256, 5),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(256, 512, 3),
            act(),
            nn.MaxPool3d(2, 2),
            ResBlock(512, 512, 3),
            act(),
            nn.Dropout(dropout),
            nn.MaxPool3d(2, 2),
            ResBlock(512, 1024, 1),
            act(),
            nn.Dropout(dropout),
            ResBlock(1024, n_out, 1),
        )
        

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = out.mean(dim=-1) 
        return out
