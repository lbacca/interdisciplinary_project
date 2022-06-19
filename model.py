import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder layer
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    def forward(self, x):
        # encode
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc3(x))
        x = F.leaky_relu(self.enc4(x))
        
        # decode
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x = F.leaky_relu(self.dec3(x))
        x = F.sigmoid(self.out(x))

        return x
    
