import librosa
from transformers import pipeline

def transcrire_audio(chemin_audio):
    print(f"1. Chargement du fichier audio : {chemin_audio}...")
    
    y, sr = librosa.load(chemin_audio, sr=16000)
    
    print("2. Chargement du modèle Whisper...")
    # On ajoute chunk_length_s=30 pour découper les longs audios
    transcripteur = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
    
    print("3. Transcription en cours (L'IA écrit ce qu'elle entend)...")
    resultat = transcripteur(y)
    
    texte_transcrit = resultat["text"]
    
    print("\n📝 --- RÉSULTAT DE LA TRANSCRIPTION ---")
    print(f'Le candidat a dit : "{texte_transcrit}"')
    print("----------------------------------------\n")
    
    return texte_transcrit

if __name__ == "__main__":
    CHEMIN = "data/test_voix.wav"
    transcrire_audio(CHEMIN)