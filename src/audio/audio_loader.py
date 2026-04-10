import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- NOTRE RECETTE SUR MESURE POUR L'AUDIO ---
class RavdessDataset(Dataset):
    
    # 1. INITIALISATION : Préparer la liste de tous les fichiers
    def __init__(self, data_dir, max_pad_len=130):
        self.data_dir = data_dir
        self.max_pad_len = max_pad_len # ~3 secondes d'audio en MFCC
        # On garde exactement le même ordre alphabétique que pour la vision !
        self.classes = ['confiance', 'degout', 'joie', 'peur', 'stress', 'surprise', 'tristesse']
        self.filepaths = []
        self.labels = []

        # On parcourt nos dossiers pour créer une grande liste des chemins de fichiers
        for label_idx, emotion in enumerate(self.classes):
            emotion_dir = os.path.join(data_dir, emotion)
            if not os.path.exists(emotion_dir): 
                continue
            
            for file in os.listdir(emotion_dir):
                if file.endswith('.wav'):
                    self.filepaths.append(os.path.join(emotion_dir, file))
                    self.labels.append(label_idx)

    # 2. LONGUEUR : Combien de fichiers on a ?
    def __len__(self):
        return len(self.filepaths)

    # 3. EXTRACTION : Le cœur du système. Que se passe-t-il quand l'IA demande 1 fichier ?
    def __getitem__(self, index):
        filepath = self.filepaths[index]
        label = self.labels[index]

        # A. Charger l'audio et extraire les MFCC (13 caractéristiques)
        y, sr = librosa.load(filepath, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # B. Gérer la taille variable (Padding / Troncature)
        if mfcc.shape[1] > self.max_pad_len:
            # Si c'est trop long, on coupe ce qui dépasse
            mfcc = mfcc[:, :self.max_pad_len] 
        else:
            # Si c'est trop court, on rajoute des zéros (du silence) à droite
            pad_width = self.max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # C. Convertir en format PyTorch (Tenseur)
        # On ajoute "unsqueeze(0)" pour dire qu'il y a 1 "canal" (comme une image en noir et blanc)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        
        return mfcc_tensor, label

# --- FONCTION POUR CRÉER LES SERVEURS DE CANTINE ---
def get_audio_loaders(base_dir='data/audio_dataset/', batch_size=32):
    print("Préparation des données audio (Extraction des MFCC)...")
    
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    train_dataset = RavdessDataset(train_dir)
    test_dataset = RavdessDataset(test_dir)
    
    # DataLoader : L'outil qui fait des paquets et mélange les données
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.classes

# Petit test rapide
if __name__ == "__main__":
    loaders = get_audio_loaders()
    print("Le chargeur audio est prêt !")