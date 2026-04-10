import torch
import torch.nn as nn
from torchvision import models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7): # Nos 7 émotions cibles : Joie, Stress, Peur, Confiance, Dégout, Tristesse, Surprise
        super(EmotionCNN, self).__init__()
        
        # 1. On charge la structure d'un ResNet18 (un modèle classique et très efficace)
        # weights=None signifie qu'on charge une coquille vide, sans "connaissances" préalables.
        self.resnet = models.resnet18(weights=None)
        
        # 2. On modifie l'entrée (la première couche convolutive appelée 'conv1')
        # On remplace le '3' (pour les couleurs RGB) par '1' (pour le noir et blanc)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. On modifie la sortie (la dernière couche linéaire appelée 'fc' pour Fully Connected)
        # On récupère le nombre de connexions entrantes de cette dernière couche...
        num_ftrs = self.resnet.fc.in_features
        # ... et on remplace la sortie par notre nombre de classes (4)
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Cette fonction définit le chemin que prend l'image (x) à travers le réseau
        return self.resnet(x)

# Petit test pour vérifier que la création du modèle fonctionne sans erreur
if __name__ == "__main__":
    model = EmotionCNN()
    print("Modèle CNN créé avec succès ! 🎉")
    print(f"La dernière couche a été modifiée pour sortir {model.resnet.fc.out_features} émotions.")