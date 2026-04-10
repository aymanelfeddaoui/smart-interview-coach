from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluer_pertinence(question, reponse_candidat):
    print("1. Chargement du 'cerveau' sémantique (Mini-RoBERTa)...")
    # On charge un modèle léger et très performant en anglais/français
    modele = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("2. Transformation des phrases en vecteurs mathématiques (Embeddings)...")
    # Le modèle transforme notre texte en listes de 384 nombres
    vecteur_question = modele.encode([question])
    vecteur_reponse = modele.encode([reponse_candidat])
    
    print("3. Calcul géométrique de l'angle entre les deux vecteurs...")
    # On calcule la similarité cosinus (le résultat est entre -1 et 1)
    score_similarite = cosine_similarity(vecteur_question, vecteur_reponse)
    
    # On convertit le score en pourcentage pour que ce soit plus lisible (0 à 100%)
    pourcentage_pertinence = round(score_similarite[0][0] * 100, 2)
    
    print("\n🧠 --- ANALYSE SÉMANTIQUE ---")
    print(f"Question du recruteur : '{question}'")
    print(f"Réponse du candidat   : '{reponse_candidat}'")
    print(f"🎯 Score de pertinence : {pourcentage_pertinence}%")
    print("----------------------------\n")
    
    return pourcentage_pertinence

# Petit test rapide
if __name__ == "__main__":
    # Test 1 : Une réponse pertinente
    q1 = "Quelles sont vos compétences en programmation ?"
    r1 = "Je maîtrise très bien le langage Python et j'utilise souvent PyTorch pour créer des algorithmes."
    evaluer_pertinence(q1, r1)
    
    # Test 2 : Un hors-sujet total
    q2 = "quelle est la difference entre developpement et Programmation?"
    r2 = "le developpement est un processus qui englobe la programmation, mais aussi la planification, la conception et le déploiement d'un projet logiciel. La programmation, quant à elle, se concentre spécifiquement sur l'écriture du code pour créer des applications ou des logiciels."
    evaluer_pertinence(q2, r2)