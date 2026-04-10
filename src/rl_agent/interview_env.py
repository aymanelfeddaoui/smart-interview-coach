import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class InterviewEnv(gym.Env):
    """
    Simulateur d'entretien d'embauche pour entraîner notre Agent IA.
    """
    def __init__(self):
        super(InterviewEnv, self).__init__()
        
        # 1. ESPACE D'ACTION (Ce que l'IA peut faire)
        # 0 = Ice-breaker, 1 = Comportementale, 2 = Technique
        self.action_space = spaces.Discrete(3)
        
        # 2. ESPACE D'OBSERVATION (Ce que l'IA voit : [Stress, Performance NLP])
        # Les valeurs vont de 0.0 (Faible) à 1.0 (Fort)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                            high=np.array([1.0, 1.0]), 
                                            dtype=np.float32)
        
        # Paramètres de l'entretien
        self.max_questions = 10
        self.current_step = 0
        
        # L'état actuel du candidat virtuel
        self.state = np.array([0.5, 0.5]) # Commence avec un stress et une perf moyens

    def reset(self, seed=None, options=None):
        """Remet l'entretien à zéro pour un nouveau candidat."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # On génère un candidat avec un stress de départ aléatoire (entre 0.3 et 0.7)
        stress_initial = random.uniform(0.3, 0.7)
        perf_initiale = 0.5
        self.state = np.array([stress_initial, perf_initiale], dtype=np.float32)
        
        # Gym demande de renvoyer l'état et un dictionnaire d'infos (vide ici)
        return self.state, {}

    def step(self, action):
        """
        C'est le cœur de la simulation. L'IA pose une question (action),
        on calcule la réaction du candidat, et on donne une note (reward).
        """
        stress, perf = self.state[0], self.state[1]
        
        # --- LOGIQUE DE SIMULATION DU CANDIDAT ---
        if action == 0: # ICE-BREAKER
            # Baisse fortement le stress, mais ne teste pas vraiment la performance
            stress = max(0.0, stress - 0.2)
            perf = min(1.0, perf + 0.05)
            
        elif action == 1: # COMPORTEMENTALE
            # Maintient le stress stable, teste la performance
            stress = min(1.0, stress + 0.05)
            perf = min(1.0, perf + random.uniform(-0.1, 0.2))
            
        elif action == 2: # TECHNIQUE
            # Augmente fortement le stress. 
            # Si le candidat est déjà trop stressé (>0.7), sa performance s'effondre !
            stress = min(1.0, stress + 0.2)
            if stress > 0.7:
                perf = max(0.0, perf - 0.3)
            else:
                perf = min(1.0, perf + random.uniform(0.0, 0.3))

        self.state = np.array([stress, perf], dtype=np.float32)
        self.current_step += 1
        
        # --- CALCUL DE LA RÉCOMPENSE (REWARD) ---
        # L'IA gagne des points si la perf est haute ET le stress est bas
        reward = perf - stress
        
        # L'IA est pénalisée si elle fait faire une crise de panique au candidat
        if stress >= 0.9:
            reward -= 2.0 

        # Vérifier si l'entretien est fini
        terminated = bool(self.current_step >= self.max_questions)
        truncated = False # Utilisé pour les limites de temps (non pertinent ici)

        return self.state, reward, terminated, truncated, {}

# Petit test du simulateur
if __name__ == "__main__":
    env = InterviewEnv()
    etat_initial, _ = env.reset()
    print(f"Début de l'entretien. État du candidat [Stress, Perf] : {etat_initial}")
    
    # On simule une IA qui pose 3 questions techniques de suite (Action 2)
    for i in range(3):
        action_ia = 2 
        nouvel_etat, recompense, fini, _, _ = env.step(action_ia)
        print(f"Question {i+1} (Technique) -> Nouvel État: {nouvel_etat}, Récompense IA: {recompense:.2f}")