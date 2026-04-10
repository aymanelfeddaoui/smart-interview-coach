import numpy as np
import random
import pickle # Pour sauvegarder notre matrice (la Q-Table)
from src.rl_agent.interview_env import InterviewEnv


def discretiser_etat(etat_continu, nb_bacs=10):
    """
    Transforme un état continu [0.45, 0.81] en indices entiers (4, 8) 
    pour pouvoir les ranger dans notre matrice (Q-Table).
    """
    stress, perf = etat_continu
    # On multiplie par 10 et on prend l'entier. Si c'est 1.0, on le force à 9.
    stress_idx = min(int(stress * nb_bacs), nb_bacs - 1)
    perf_idx = min(int(perf * nb_bacs), nb_bacs - 1)
    return (stress_idx, perf_idx)

def train_q_learning(episodes=100000): 
    print(f"Lancement de l'entraînement Q-Learning pour {episodes} entretiens...")
    env = InterviewEnv()
    
    # 1. INITIALISATION DE LA Q-TABLE
    # Matrice de taille (10 niveaux de stress, 10 niveaux de perf, 3 actions possibles)
    # Remplie de zéros au départ.
    q_table = np.zeros((10, 10, 3))
    
    # HYPERPARAMÈTRES MATHÉMATIQUES
    alpha = 0.1       # Taux d'apprentissage (Learning rate)
    gamma = 0.9       # Facteur de réduction (Importance des récompenses futures)
    epsilon = 1.0     # Taux d'exploration initial (100% de hasard)
    epsilon_min = 0.1 # On garde toujours 10% de hasard pour éviter d'être bloqué
    epsilon_decay = 0.995 # Vitesse à laquelle l'IA arrête d'explorer au hasard
    
    # 2. LA BOUCLE D'ENTRAÎNEMENT
    for episode in range(episodes):
        etat_continu, _ = env.reset()
        etat = discretiser_etat(etat_continu)
        fini = False
        
        while not fini:
            # A. CHOIX DE L'ACTION (Exploration vs Exploitation)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Hasard (Exploration)
            else:
                action = np.argmax(q_table[etat[0], etat[1]]) # Meilleur choix connu (Exploitation)
                
            # B. L'IA FAIT L'ACTION DANS LE SIMULATEUR
            prochain_etat_continu, recompense, fini, _, _ = env.step(action)
            prochain_etat = discretiser_etat(prochain_etat_continu)
            
            # C. MISE À JOUR DE LA MATRICE (L'Équation de Bellman simplifiée)
            # Ancienne valeur
            ancienne_valeur = q_table[etat[0], etat[1], action]
            # La meilleure valeur possible à l'étape suivante
            max_valeur_future = np.max(q_table[prochain_etat[0], prochain_etat[1]])
            
            # On calcule la nouvelle valeur en mixant l'ancienne et la nouvelle découverte
            nouvelle_valeur = ancienne_valeur + alpha * (recompense + gamma * max_valeur_future - ancienne_valeur)
            q_table[etat[0], etat[1], action] = nouvelle_valeur
            
            # On avance dans le temps
            etat = prochain_etat
            
        # D. DIMINUTION DE L'EXPLORATION
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if episode % 1000 == 0:
            print(f"Entretien n°{episode} terminé. Epsilon actuel : {epsilon:.2f}")

    print("\nEntraînement terminé ! Le recruteur IA est formé. 🧠")
    
    # 3. SAUVEGARDE DU CERVEAU (La matrice Q-Table)
    with open('models/q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    print("La Q-Table a été sauvegardée dans 'models/q_table.pkl'")

if __name__ == "__main__":
    train_q_learning()