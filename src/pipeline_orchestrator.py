import numpy as np
import pickle

# On importe NOS PROPRES fonctions créées dans les phases précédentes !
from src.nlp.speech_to_text import transcrire_audio
from src.nlp.analyze_text import evaluer_pertinence
from src.rl_agent.q_learning_agent import discretiser_etat

# Dictionnaire pour traduire la décision finale de l'IA
ACTION_MAP = {
    0: "🧊 ICE-BREAKER",
    1: "🤝 QUESTION COMPORTEMENTALE",
    2: "⚙️ QUESTION TECHNIQUE"
}

def executer_tour_entretien(chemin_audio, question_posee, emotion_detectee):
    print("\n" + "="*50)
    print(" 🔄 LANCEMENT DU PIPELINE D'ANALYSE")
    print("="*50)

    # --- ÉTAPE 1 : PERCEPTION (Définir le Stress) ---
    print(f"👁️/👂 Perception : L'émotion détectée par les capteurs est '{emotion_detectee.upper()}'")
    if emotion_detectee in ["stress", "peur", "tristesse"]:
        stress_score = 0.8  # Stress élevé
    elif emotion_detectee in ["joie", "confiance"]:
        stress_score = 0.2  # Stress faible
    else:
        stress_score = 0.5  # Neutre

    # --- ÉTAPE 2 : COGNITION (Définir la Performance) ---
    print("📝 Cognition : L'IA écoute et écrit la réponse...")
    texte_candidat = transcrire_audio(chemin_audio)
    
    print("🧠 Cognition : L'IA évalue la pertinence de la réponse...")
    perf_score_100 = evaluer_pertinence(question_posee, texte_candidat)
    perf_score = perf_score_100 / 100.0  # Ramener entre 0.0 et 1.0

    # --- ÉTAPE 3 : DÉCISION (Reinforcement Learning) ---
    print("\n⚖️ Décision : Le cerveau RL analyse la situation...")
    etat_continu = (stress_score, perf_score)
    etat_discret = discretiser_etat(etat_continu)

    # Chargement de la Q-Table (l'expérience de l'IA)
    with open('models/q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)

    # L'IA choisit la meilleure action
    action = np.argmax(q_table[etat_discret[0], etat_discret[1]])
    
    print("\n" + "="*50)
    print(" 🎯 RÉSULTAT DU TOUR")
    print("="*50)
    print(f"Niveau de Stress      : {stress_score*100:.0f}%")
    print(f"Niveau de Performance : {perf_score*100:.0f}%")
    print(f"-> Prochaine action de l'IA : Poser une {ACTION_MAP[action]}")
    print("="*50 + "\n")

    return action

# Test de notre système nerveux central
if __name__ == "__main__":
    AUDIO_TEST = "data/scripte2.wav"
    QUESTION_TEST = "Parlez-moi de votre expérience avec le langage Python."
    EMOTION_TEST = "stress" # On simule que le module audio a détecté du stress
    
    executer_tour_entretien(AUDIO_TEST, QUESTION_TEST, EMOTION_TEST)