import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Charger le fichier audio
chemin_audio = 'data/test_audio.wav'
print(f"Chargement de l'audio : {chemin_audio}...")

# librosa.load renvoie deux choses :
# 'y' : un grand tableau (numpy array) contenant les amplitudes du son
# 'sr' : la fréquence d'échantillonnage (combien de mesures par seconde)
y, sr = librosa.load(chemin_audio, sr=22050)

print(f"L'audio dure {len(y)/sr:.2f} secondes.")
print(f"L'ordinateur voit un tableau de {len(y)} nombres.")

# 2. Extraire l'ADN de la voix : les MFCC
# On demande à librosa de calculer 13 caractéristiques MFCC standard
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print(f"Taille de la matrice MFCC extraite : {mfccs.shape}")

# 3. Visualiser le résultat
plt.figure(figsize=(10, 4))
# On affiche les MFCC sous forme de carte de chaleur (spectrogramme)
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme des MFCC (Ce que l\'IA va "voir")')
plt.tight_layout()
plt.show()