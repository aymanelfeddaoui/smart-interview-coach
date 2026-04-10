import pandas as pd
import numpy as np
import cv2
import os
import random

CSV_PATH = 'data/fer2013_raw/train.csv'
BASE_DIR = 'data/'

# On cartographie maintenant les 7 émotions du dataset FER-2013
EMOTION_MAP = {
    0: 'stress',     # Angry (utilisé pour le stress/tension)
    1: 'degout',     # Disgust
    2: 'peur',       # Fear
    3: 'joie',       # Happy
    4: 'tristesse',  # Sad
    5: 'surprise',   # Surprise
    6: 'confiance'   # Neutral (utilisé pour la confiance/calme)
}

def create_dirs():
    """Crée l'arborescence complète"""
    for split in ['train', 'test']:
        for emotion in EMOTION_MAP.values():
            os.makedirs(os.path.join(BASE_DIR, split, emotion), exist_ok=True)

def process_data():
    print(f"Lecture du fichier {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    counts = {'train': 0, 'test': 0}

    for index, row in df.iterrows():
        emotion_id = int(row['emotion'])
        emotion_name = EMOTION_MAP[emotion_id]
        
        # Transformation des pixels
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
        image = pixels.reshape(48, 48)
        
        # Répartition 80% Train / 20% Test
        split = 'train' if random.random() < 0.8 else 'test'
        
        # Sauvegarde
        filename = f"{split}_{emotion_name}_{index}.jpg"
        filepath = os.path.join(BASE_DIR, split, emotion_name, filename)
        cv2.imwrite(filepath, image)
        
        counts[split] += 1
        
        if index % 5000 == 0 and index > 0:
            print(f"{index} lignes traitées...")

    print("\nExtraction complète terminée ! 🎉")
    print(f"Images d'entraînement : {counts['train']}")
    print(f"Images de test : {counts['test']}")
    print(f"Total des images extraites : {counts['train'] + counts['test']}")

if __name__ == "__main__":
    create_dirs()
    process_data()