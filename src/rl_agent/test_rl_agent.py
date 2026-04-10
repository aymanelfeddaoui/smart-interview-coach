import pickle
import numpy as np

# On importe le simulateur et notre outil de traduction d'état
from src.rl_agent.interview_env import InterviewEnv
from src.rl_agent.q_learning_agent import discretiser_etat

# Petit dictionnaire pour traduire le choix mathématique de l'IA en texte lisible
ACTION_MAP = {
    0: "🧊 ICE-BREAKER (Pour détendre l'atmosphère)",
    1: "🤝 QUESTION COMPORTEMENTALE (Pour tester les Soft-Skills)",
    2: "⚙️ QUESTION TECHNIQUE (Pour tester les Hard-Skills)"
}

def tester_recruteur_ia():
    print("1. Réveil de l'Intelligence Artificielle...")
    
    # A. Chargement de la matrice (La Q-Table)
    try:
        with open('models/q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print("❌ Erreur : Le fichier 'models/q_table.pkl' est introuvable.")
        return

    print("2. Entrée du candidat virtuel dans la salle d'entretien...")
    env = InterviewEnv()
    etat_continu, _ = env.reset()
    fini = False
    numero_question = 1

    print("\n" + "="*50)
    print(" DÉBUT DE L'ENTRETIEN DYNAMIQUE")
    print("="*50)

    # La boucle de l'entretien (max 10 questions, géré par l'environnement)
    while not fini:
        # B. Traduire l'état continu en index de matrice
        etat = discretiser_etat(etat_continu)
        
        # Pour un affichage plus humain (en pourcentages)
        stress_reel = etat_continu[0] * 100
        perf_reelle = etat_continu[1] * 100

        print(f"\n📊 TOUR {numero_question} | État du candidat -> Stress : {stress_reel:.0f}% | Performance : {perf_reelle:.0f}%")

        # C. L'IA PREND SA DÉCISION (Politique Gloutonne / Greedy)
        # On regarde la ligne correspondant à l'état, et on prend l'index de la valeur Max
        action = np.argmax(q_table[etat[0], etat[1]])
        
        print(f"🤖 Décision de l'IA Coach -> Je pose une {ACTION_MAP[action]}")

        # D. On observe la réaction du candidat
        etat_continu, recompense, fini, _, _ = env.step(action)
        numero_question += 1

    print("\n" + "="*50)
    print(" FIN DE L'ENTRETIEN")
    print("="*50)
    print("L'IA a terminé son évaluation en adaptant son parcours ! 🎉")

if __name__ == "__main__":
    tester_recruteur_ia()