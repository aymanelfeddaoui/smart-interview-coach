import sounddevice as sd
from scipy.io.wavfile import write

def enregistrer_microphone(duree_secondes=6, chemin_sortie="data/live_reponse.wav"):
    """
    Ouvre le microphone, enregistre la voix du candidat, 
    et sauvegarde le fichier au format attendu par Whisper.
    """
    fs = 16000  # Fréquence exigée par notre modèle Speech-to-Text
    
    print(f"\n🎤 [MICROPHONE OUVERT] Parlez maintenant ! ({duree_secondes} secondes)...")
    
    # Lancement de l'enregistrement (1 canal = mono)
    enregistrement = sd.rec(int(duree_secondes * fs), samplerate=fs, channels=1, dtype='int16')
    
    # On bloque le programme jusqu'à la fin des 6 secondes
    sd.wait() 
    
    print("✅ [MICROPHONE FERMÉ] Enregistrement terminé.")
    
    # Sauvegarde sur le disque dur
    write(chemin_sortie, fs, enregistrement)
    
    return chemin_sortie

if __name__ == "__main__":
    # Petit test rapide
    fichier = enregistrer_microphone(duree_secondes=3)
    print(f"Fichier sauvegardé ici : {fichier}")