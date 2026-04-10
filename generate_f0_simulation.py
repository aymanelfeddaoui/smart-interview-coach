import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Configurer la police pour un rendu professionnel (style Arial/Helvetica)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans']
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10

# 1. Configuration temporelle (5 secondes)
t = np.linspace(0, 5, 500) # 500 points pour la fluidité

# 2. Simulation des Contours F0 (Fréquence Fondamentale en Hz)
# Baseline gender-neutral (Moyenne autour de 160 Hz)

# --- Faible Stress (Ligne Bleue) ---
# Smoother, variations lentes de prosodie naturelle.
baseline_low = 160
variation_low = 15 * np.sin(t * 1.5) # Prosodie naturelle lente
jitter_low = np.random.normal(0, 1.5, len(t)) # Très faible jitter (instabilité)
f0_low = baseline_low + variation_low + jitter_low

# --- Haut Stress (Ligne Rouge) ---
# Baseline plus haute, variations rapides (shimmer), jitter plus fort.
# Le stress tend à contracter les cordes vocales -> F0 plus aiguë.
baseline_high = 210 # +50Hz de décalage lié au stress
variation_high = 25 * np.sin(t * 2.5) # Variations prosodiques plus rapides/instables
jitter_high = np.random.normal(0, 5.0, len(t)) # Jitter (micro-instabilités) plus fort
f0_high = baseline_high + variation_high + jitter_high

# 3. Lissage des courbes (Moving Average) pour un rendu "extracted features"
def smooth_curve(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Fenêtre de lissage (ajustable pour le réalisme visuel)
window = 25 
f0_low_smooth = smooth_curve(f0_low, window)
f0_high_smooth = smooth_curve(f0_high, window)

# 4. Création du graphique professionnel
plt.figure(figsize=(10, 5))

# Tracer les contours lissés avec les bonnes couleurs
# On commence après la fenêtre de lissage pour éviter les artefacts de bord
start = window // 2
end = - (window // 2)
plt.plot(t[start:end], f0_low_smooth[start:end], label='Faible Stress', color='royalblue', linewidth=2.5, linestyle='-')
plt.plot(t[start:end], f0_high_smooth[start:end], label='Haut Stress', color='firebrick', linewidth=2.5, linestyle='--')

# Ajout de la zone ombrée de variation (optionnel mais très pro pour visualiser l'instabilité)
# plt.fill_between(t, f0_low_smooth - 5, f0_low_smooth + 5, color='blue', alpha=0.1)
# plt.fill_between(t, f0_high_smooth - 10, f0_high_smooth + 10, color='red', alpha=0.1)

# Formatage des axes et légende
plt.title('Comparaison du Contour de Fréquence Fondamentale (F0)')
plt.xlabel('Temps (secondes)')
plt.ylabel('F0 (Hz)')
plt.legend(loc='upper right', frameon=True, fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 5. Sauvegarde en Haute Qualité (300 dpi pour le rapport final)
nom_image = "f0_contour_comparison.png"
plt.savefig(nom_image, dpi=300, bbox_inches='tight')
print(f"✅ Graphique généré et sauvegardé sous : {nom_image}")

# Afficher pour vérification immédiate
plt.show()