import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    # 1. INITIALISATION : On définit les "pièces" de notre moteur
    def __init__(self, input_size=13, hidden_size=64, num_classes=7):
        super(AudioLSTM, self).__init__()
        
        # input_size=13 car nous avons extrait 13 caractéristiques MFCC
        # hidden_size=64 est la "taille" de la mémoire du LSTM (réglable)
        # batch_first=True dit à PyTorch que nos paquets commencent par la dimension Batch
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # La couche finale qui transforme la mémoire de 64 nombres en 7 probabilités (nos 7 émotions)
        self.fc = nn.Linear(hidden_size, num_classes)

    # 2. LE FLUX (FORWARD) : Le chemin que parcourt la donnée
    def forward(self, x):
        # A. Format actuel de 'x' : (Batch, Canal=1, Features=13, Temps=130)
        
        # On supprime la dimension du Canal (qui ne sert à rien ici)
        x = x.squeeze(1) # Résultat : (Batch, 13, 130)
        
        # Le LSTM veut le Temps avant les Features. On inverse (transpose) les dimensions 1 et 2
        x = x.transpose(1, 2) # Résultat : (Batch, Temps=130, Features=13)
        
        # B. On passe la donnée dans le LSTM
        # out contient les prédictions à CHAQUE instant t.
        out, _ = self.lstm(x)
        
        # C. On ne s'intéresse qu'à la prédiction de la TOUTE DERNIÈRE étape de temps
        # (quand le modèle a fini d'écouter tout l'audio)
        out = out[:, -1, :] 
        
        # D. On prend la décision finale
        out = self.fc(out)
        
        return out

# Petit test rapide pour valider l'architecture
if __name__ == "__main__":
    # On crée une fausse donnée pour simuler la sortie de notre DataLoader
    # Batch=1, Canal=1, Features=13, Temps=130
    donnee_fictive = torch.randn(1, 1, 13, 130) 
    
    # On crée le modèle
    modele = AudioLSTM()
    
    # On fait passer la donnée
    prediction = modele(donnee_fictive)
    
    print("Architecture LSTM valide !")
    print(f"La prédiction finale a pour forme : {prediction.shape} (1 donnée, 7 émotions)")