import os
import shutil
import random

SOURCE_DIR = 'data/ravdess_raw/'
BASE_AUDIO_DIR = 'data/audio_dataset/'

RAVDESS_MAP = {
    '01': 'confiance', '02': 'confiance', '03': 'joie',
    '04': 'tristesse', '05': 'stress', '06': 'peur',
    '07': 'degout', '08': 'surprise'
}

def create_audio_dirs():
    emotions_uniques = set(RAVDESS_MAP.values())
    for split in ['train', 'test']:
        for emotion in emotions_uniques:
            os.makedirs(os.path.join(BASE_AUDIO_DIR, split, emotion), exist_ok=True)

def process_ravdess():
    print(f"Exploration approfondie du dossier {SOURCE_DIR}...")
    compteurs = {'train': 0, 'test': 0}
    fichiers_trouves = 0

    # --- LA MAGIE DE OS.WALK ---
    # Cette boucle "marche" à travers tous les sous-dossiers automatiquement !
    for dossier_actuel, sous_dossiers, fichiers in os.walk(SOURCE_DIR):
        for fichier in fichiers:
            fichiers_trouves += 1
            
            # On ignore ce qui n'est pas un fichier audio .wav
            if not fichier.endswith('.wav'):
                continue
                
            nom_sans_extension = fichier.replace('.wav', '')
            parties = nom_sans_extension.split('-')
            
            if len(parties) != 7:
                continue
                
            # L'émotion est à l'index 2
            emotion_code = parties[2]
            
            if emotion_code in RAVDESS_MAP:
                emotion_name = RAVDESS_MAP[emotion_code]
                split = 'train' if random.random() < 0.8 else 'test'
                
                # Attention ici : on utilise "dossier_actuel" pour récupérer le fichier là où il est !
                chemin_origine = os.path.join(dossier_actuel, fichier)
                chemin_destination = os.path.join(BASE_AUDIO_DIR, split, emotion_name, fichier)
                
                shutil.copy2(chemin_origine, chemin_destination)
                compteurs[split] += 1

    print(f"\nTotal des fichiers vus (tous formats confondus) : {fichiers_trouves}")
    print("\nTri audio terminé ! 🎙️")
    print(f"Fichiers pour l'entraînement : {compteurs['train']}")
    print(f"Fichiers pour le test : {compteurs['test']}")

if __name__ == "__main__":
    if os.path.exists(SOURCE_DIR):
        create_audio_dirs()
        process_ravdess()
    else:
        print(f"❌ Erreur : Le dossier {SOURCE_DIR} n'existe pas.")