import torch
import torch.nn as nn

# Paramètres de configuration
img_size = 64        # Taille de l'image générée (64x64)
z_dim = 100          # Dimension du vecteur latent
channels = 3         # Nombre de canaux (RGB)
feature_maps = 64    # Facteur multiplicatif pour les filtres

# Générateur DCGAN
class Generator(nn.Module):
    def __init__(self, z_dim, channels, feature_maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Entrée : vecteur latent de taille (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # État intermédiaire : (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # État intermédiaire : (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # État intermédiaire : (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # État intermédiaire : (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Sortie : image de taille (channels) x 64 x 64
        )

    def forward(self, z):
        return self.net(z)

# Discriminateur DCGAN
class Discriminator(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Entrée : image de taille (channels) x 64 x 64
            nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # État intermédiaire : (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # État intermédiaire : (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # État intermédiaire : (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Réduction à un scalaire
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Sortie : (1) (aplatie en vecteur)
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)
