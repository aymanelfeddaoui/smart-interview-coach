import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuration
# Utilise un vrai fichier de ton projet (ex: celui enregistré par ton micro)
chemin_audio = "data/live_reponse.wav" 
duree_max = 5.0 # On limite à 5 secondes comme demandé dans ta légende
n_mfcc = 13     # Les 13 coefficients classiques pour la voix

print("Analyse de l'audio en cours...")

# 2. Chargement de l'audio (16kHz pour être cohérent avec ton LSTM et Whisper)
y, sr = librosa.load(chemin_audio, sr=16000, duration=duree_max)

# 3. Calcul des MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

# 4. Création du graphique professionnel
plt.figure(figsize=(10, 4))
# librosa.display met tout en forme automatiquement (axes, temps)
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')

plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme MFCC (13 coefficients)')
plt.xlabel('Temps (secondes)')
plt.ylabel('Coefficients MFCC')
plt.tight_layout()

# 5. Sauvegarde en Haute Qualité (300 dpi est le standard pour les rapports imprimés)
nom_image = "mfcc_spectrogram.png"
plt.savefig(nom_image, dpi=300, bbox_inches='tight')
print(f"✅ Graphique généré et sauvegardé sous : {nom_image}")

# Afficher l'image pour vérifier
plt.show()