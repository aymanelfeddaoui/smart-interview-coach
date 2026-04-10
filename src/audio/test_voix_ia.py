import torch
import librosa
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# On importe l'architecture de notre modèle LSTM
from src.audio.audio_model import AudioLSTM

# --- CONFIGURATION ---
CLASSES = ['confiance', 'degout', 'joie', 'peur', 'stress', 'surprise', 'tristesse']
MAX_PAD_LEN = 130 # Exactement la même taille qu'à l'entraînement !
CHEMIN_AUDIO = 'data/test_voix.wav'

def evaluer_voix(audio_path):
    print("1. Chargement du 'cerveau' vocal...")
    # On recrée la coquille vide
    model = AudioLSTM(num_classes=7)
    # On charge les connaissances sauvegardées
    model.load_state_dict(torch.load('models/audio_model_v1.pth'))
    # On verrouille le modèle en mode "examen"
    model.eval() 

    print(f"2. Préparation mathématique de l'audio : {audio_path}")
    # A. Lecture de l'audio et extraction des MFCC
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # B. Application de la règle stricte (Troncature ou Padding)
    if mfcc.shape[1] > MAX_PAD_LEN:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    else:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # C. Conversion en Tenseur PyTorch
    # Le modèle attend (Batch, Canal, Features, Temps).
    # Notre MFCC est (13, 130). 
    # unsqueeze(0) ajoute la dimension Batch -> (1, 13, 130)
    # un deuxième unsqueeze(0) (ou 1) ajoute la dimension Canal -> (1, 1, 13, 130)
    tensor_audio = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    print("3. Inférence (L'IA écoute et réfléchit)...")
    # On bloque le calcul des gradients (économise de la mémoire)
    with torch.no_grad():
        # L'IA rend son verdict (7 probabilités)
        outputs = model(tensor_audio)
        
        # On cherche l'index de la probabilité la plus forte
        _, predicted = torch.max(outputs, 1)
        index_gagnant = predicted.item()
        
        # On traduit l'index en mot grâce à notre liste CLASSES
        emotion = CLASSES[index_gagnant]

    print(f"\n🎤 Résultat : L'Intelligence Artificielle détecte l'émotion -> {emotion.upper()} !")

if __name__ == "__main__":
    evaluer_voix(CHEMIN_AUDIO)