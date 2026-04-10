# Smart Interview Coach — Description technique complète

**Titre du projet :** Smart Interview Coach — Architecture d’une IA multimodale et décisionnelle  
**Institution :** ENSIAS  
**Document :** description détaillée de bout en bout (alignée sur la présentation Beamer et sur l’architecture fonctionnelle du système)  
**Année de référence :** 2026  

---

## Table des matières

1. [Vue d’ensemble](#1-vue-densemble)  
2. [Problématique métier et scientifique](#2-problématique-métier-et-scientifique)  
3. [Concept de la solution](#3-concept-de-la-solution)  
4. [Architecture globale et pipeline end-to-end](#4-architecture-globale-et-pipeline-end-to-end)  
5. [Fusion multimodale](#5-fusion-multimodale)  
6. [Branche vision : théorie](#6-branche-vision--théorie)  
7. [Branche vision : pratique (PyTorch, OpenCV)](#7-branche-vision--pratique-pytorch-opencv)  
8. [Branche audio : théorie (MFCC, LSTM)](#8-branche-audio--théorie-mfcc-lstm)  
9. [Branche audio : pratique (Librosa, Sounddevice)](#9-branche-audio--pratique-librosa-sounddevice)  
10. [Speech-to-text avec Whisper](#10-speech-to-text-avec-whisper)  
11. [Cognition langagière : Transformers, RoBERTa, MiniLM](#11-cognition-langagière--transformers-roberta-minilm)  
12. [Scoring par similarité cosinus](#12-scoring-par-similarité-cosinus)  
13. [Cerveau décisionnel : formalisation MDP](#13-cerveau-décisionnel--formalisation-mdp)  
14. [Q-learning, équation de Bellman, exploration](#14-q-learning-équation-de-bellman-exploration)  
15. [Entraînement et convergence en pratique](#15-entraînement-et-convergence-en-pratique)  
16. [Interface web : Streamlit et machine d’états](#16-interface-web--streamlit-et-machine-détats)  
17. [Temps réel : vidéo, audio, navigateur](#17-temps-réel--vidéo-audio-navigateur)  
18. [Scénarios : candidat stressé vs confiant](#18-scénarios--candidat-stressé-vs-confiant)  
19. [MLOps, traçabilité et qualité](#19-mlops-traçabilité-et-qualité)  
20. [Synthèse des métriques du projet](#20-synthèse-des-métriques-du-projet)  
21. [Perspectives](#21-perspectives)  
22. [Glossaire](#22-glossaire)  
23. [Correspondance avec les figures de la présentation](#23-correspondance-avec-les-figures-de-la-présentation)  

---

## 1. Vue d’ensemble

Le **Smart Interview Coach** est un système logiciel conçu pour **simuler, accompagner et évaluer** un entretien d’embauche ou d’admission de manière **multimodale** : il exploite simultanément la **vidéo** (expressions, posture faciale), l’**audio** (prosodie, intonation, indicateurs liés au stress vocal) et le **texte** (transcription de la parole puis analyse sémantique). Une couche **décisionnelle** basée sur les principes de l’**apprentissage par renforcement** (processus de décision markovien et Q-learning sur espace d’états discrétisé) choisit le **type de question** ou d’**intervention** (ice-breaker, question technique, mise au défi sur les soft skills) en fonction de l’état estimé du candidat (stress, performance).

L’objectif scientifique et pédagogique est double :

- **Pour la recherche / l’ingénierie :** montrer une chaîne **perception → fusion → langage → décision** cohérente, avec des **métriques quantifiables** à chaque étage.  
- **Pour l’utilisateur :** fournir un **feedback structuré** et un **déroulement adaptatif**, souvent absent des entretiens purement humains ou des outils textuels seuls.

Ce document enchaîne **du contexte métier** jusqu’aux **détails d’implémentation** (bibliothèques, fréquences d’échantillonnage, ordres de grandeur de latence et de précision), en restant aligné sur les chiffres et la structure déjà présentés dans la présentation Beamer du projet.

---

## 2. Problématique métier et scientifique

### 2.1 Limites des entretiens « classiques »

Les entretiens traditionnels reposent fortement sur **l’appréciation humaine**. Celle-ci, même chez des professionnels expérimentés, peut être affectée par des **biais** (similarité, ordre des entretiens, ancres, fatigue) et par la **difficulté à objectiver** des dimensions comme le **stress** ou la **cohérence** entre ce qui est dit et ce que suggèrent les signaux non verbaux.

Du point de vue **traçabilité**, il est rare d’avoir :

- une **séquence temporelle** des signaux (voix, visage) **horodatée** et réexploitable pour l’analyse ;  
- des **scores intermédiaires** (qualité de réponse, niveau de stress vocal) **stockés de manière systématique** au fil de l’entretien ;  
- un **retour immédiat** au candidat sous forme de tableaux de bord ou de synthèses chiffrées.

### 2.2 Côté candidat

Pour le candidat, le **stress physiologique** se manifeste souvent par la **voix** (débit, hauteur fondamentale, micro-tremblements) et le **regard / expression** (fixation, sourire forcé, évitement du contact visuel). Dans un entretien classique, ces signaux sont interprétés **qualitativement** ; ils ne sont en général **pas utilisés comme entrée** d’un système qui **adapte** le déroulé (ralentir, simplifier, ou au contraire approfondir techniquement).

Par ailleurs, **s’entraîner** en conditions réalistes (caméra, micro, chronomètre, enchaînement de questions) est coûteux en organisation. Un **coach logiciel** peut reproduire une boucle **question – réponse – feedback** de façon **répétable** et **paramétrable**.

### 2.3 Enjeu pour le Smart Interview Coach

Le projet vise à combler ces lacunes en :

1. **capturant** plusieurs modalités de façon synchronisée ;  
2. **inférant** un état (stress, performance langagière, etc.) de façon **automatisée** ;  
3. **décide** de la prochaine action pédagogique via une politique **apprise ou calibrée** (RL) ;  
4. **présentant** le tout dans une **interface web** compréhensible (Streamlit).

---

## 3. Concept de la solution

### 3.1 Les trois piliers « écouter, regarder, s’adapter »

- **Écouter :** enregistrement audio **16 kHz**, traitement **MFCC** / énergie / \(F_0\), et éventuellement modèles **séquentiels (LSTM)** sur les séries temporelles acoustiques.  
- **Regarder :** flux **vidéo**, **détection faciale** (OpenCV), **réseau convolutif** (type ResNet ou équivalent) pour des attributs liés à l’expression ou à l’attention.  
- **S’adapter :** le module **RL** met à jour une représentation d’**état** (stress, performance) et choisit une **action** parmi ice-breaker, question technique, focus soft skills.

### 3.2 Boucle fermée perception → cognition → action

1. **Perception :** images + sons + (après transcription) texte.  
2. **Cognition :** embeddings, scores de similarité, agrégats multimodaux.  
3. **Action :** type de prochaine consigne ou question, cohérent avec la politique \( \pi(s) \) ou avec une variante \(\varepsilon\)-gloutonne issue de **Q(s,a)**.

Cette boucle est le **cœur architectural** du Smart Interview Coach : elle justifie le qualificatif **« décisionnel »** en plus de **« multimodal »**.

---

## 4. Architecture globale et pipeline end-to-end

### 4.1 Les cinq étapes du pipeline (rappel structurant)

1. **Acquisition**  
   Flux **vidéo** et **audio** synchronisés. Côté implémentation typique : accès caméra via OpenCV ou pont navigateur ; entrée micro via **Sounddevice** ; horodatage commun pour l’alignement des modalités.

2. **Prétraitement**  
   - Vidéo : **détection et recadrage** du visage, éventuellement resize fixe pour le CNN.  
   - Audio : fenêtrage, **FFT courte**, **coefficients MFCC**, normalisation de niveau ; **filtrage passe-bande** sur la bande parole si besoin.

3. **Fusion multimodale**  
   Combinaison des **vecteurs de caractéristiques** ou des **scores scalaires** issus de chaque branche (image, audio, texte une fois disponible). Voir section 5.

4. **Cognition**  
   - **Whisper** transcrit l’audio en texte.  
   - Un encodeur type **MiniLM / RoBERTa** produit des **embeddings** pour comparer la réponse du candidat à une **réponse de référence** (similarité cosinus, score 0–100).

5. **Décision RL**  
   Mise à jour de l’état **(Stress, Perf)** et choix d’**actions** pédagogiques selon la **Q-table** (ou politique dérivée) apprise sur des **entretiens virtuels**.

### 4.2 Schéma logique

Dans une documentation ou une slide, ce pipeline est souvent représenté par un **schéma linéaire** avec des flèches : Acquisition → Prétraitement → Fusion → (Transcription + NLP) → Décision, avec des **retours** possibles (par exemple, mise à jour de l’état RL après chaque tour de parole). Le fichier image prévu côté présentation : `figures/schema_pipeline_end_to_end.png`.

---

## 5. Fusion multimodale

### 5.1 Principe

Pour chaque **instant** ou **segment temporel**, on dispose de :

- un **vecteur image** (sortie couche fully-connected d’un CNN, ou embedding spécialisé) ;  
- un **vecteur audio** (statistiques sur MFCC, sortie LSTM, ou les deux) ;  
- après latence de transcription, un **vecteur texte** (embedding de phrase).

La **fusion** consiste à produire un **vecteur joint** ou un **tuple de scores** exploitables par la suite (état RL, affichage dashboard). Stratégies classiques :

- **Concaténation** simple suivie d’un petit **MLP** (perceptron multicouche) si on dispose de données étiquetées pour l’état global ;  
- **Pondération fixe** ou **apprentissage de poids** par modalité (attention ou gating) ;  
- **Règles calibrées** (pour prototypage) : par exemple combinaison linéaire de scores de stress vision et audio avec coefficients issus d’une régression sur données annotées.

### 5.2 Alignement temporel

La **parole** et le **mouvement des lèvres** doivent rester **cohérents** pour une interprétation humaine et pour certaines métriques. Le projet vise une **synchronisation audio–vidéo** de l’ordre de **± 40 ms** en banc de test interne. Un **horodatage unique** (timestamp monotonique par frame audio/vidéo) est indispensable.

### 5.3 Latence cible

Entre l’arrivée d’une **frame vidéo** (ou d’un bloc audio) et une **décision locale** affichable (par exemple mise à jour d’un indicateur ou choix d’action), la latence cible du projet se situe dans une fourchette **300–800 ms** selon la charge GPU/CPU et le mode batch ou streaming des modèles lourds (Whisper, gros encodeur NLP).

---

## 6. Branche vision : théorie

### 6.1 Réseaux convolutifs (CNN)

Les **CNN** appliquent des **filtres convolutifs** locaux sur l’image : chaque couche agrège des informations de voisinage spatial. Les couches profondes captent des hiérarchies **bords → textures → parties du visage → concepts** utiles à la classification (émotion, fatigue, attention).

### 6.2 ResNet et connexions résiduelles

Les **réseaux résiduels (ResNet)** introduisent des **sauts d’identité** : la sortie d’un bloc est \( y = F(x) + x \), où \(F\) est le bloc non-linéaire à apprendre. En pratique, cela facilite l’**optimisation** des très grands réseaux en atténuant le problème du **gradient qui s’évanouit** et en permettant d’empiler beaucoup de couches.

Une forme schématique de bloc :

\[
h^{(\ell+1)} = \sigma\bigl( W^{(\ell)} h^{(\ell)} + b^{(\ell)} + h^{(\ell)} \bigr)
\]

où \(\sigma\) est une non-linéarité (ReLU, GELU, etc. selon l’architecture).

### 6.3 Data augmentation

Une **augmentation** du jeu d’entraînement (**flips** horizontaux si acceptable pour les visages, **rotations légères**, variations de **luminosité/contraste**) **augmente artificiellement** la diversité des exemples vus par le modèle. Effet recherché : **meilleure généralisation** et **réduction du sur-apprentissage** sur les détails non pertinents du jeu d’origine.

### 6.4 Dropout

Le **dropout** met à zéro aléatoirement une fraction de neurones pendant l’entraînement. Cela force le réseau à ne pas **sur-dépendre** de quelques neurones et agit comme **régularisation**. En inférence, on utilise en général le réseau **complet** avec des poids parfois **rescalés** (ou dropout désactivé) selon l’implémentation.

---

## 7. Branche vision : pratique (PyTorch, OpenCV)

### 7.1 Chaîne de traitement

1. **Capture** du flux vidéo.  
2. **Détection faciale** OpenCV (cascade Haar historique, ou **réseau léger** plus récent selon le dépôt).  
3. **Recadrage** et **redimensionnement** à la taille d’entrée du CNN.  
4. **Inférence** PyTorch : forward pass, logits ou probabilités par classe.  
5. **Entraînement** : boucles train/validation, **early stopping** sur métrique validation, suivi de la **loss** (cross-entropy typiquement).

### 7.2 Résultats quantitatifs (jeu de validation du projet)

| Indicateur | Valeur |
|-----------|--------|
| Accuracy (validation) | **87,72 %** |
| F1 macro (classes émotion / état) | **0,84** (ordre de grandeur type benchmark interne) |
| Temps d’inférence | **12–18 ms** par image sur GPU milieu de gamme |

Ces chiffres permettent d’argumenter qu’une **inférence rapide** est compatible avec une boucle **temps réel** modeste (quelques dizaines de FPS théoriques côté modèle seul, le reste étant consommé par capture, affichage et autres pipelines).

### 7.3 Visualisations attendues

Pour la communication scientifique : courbes **loss / accuracy**, **matrice de confusion**, éventuellement **courbes ROC** par classe. Fichier prévu : `figures/courbe_loss_accuracy_modele_vision.png`.

---

## 8. Branche audio : théorie (MFCC, LSTM)

### 8.1 Du signal brut au spectrogramme

Le signal audio est **échantillonné** (ici **16 kHz**). On le découpe en **fenêtres** qui se chevauchent (overlap). Sur chaque fenêtre, une **transformée de Fourier** courte fournit un **spectre** ; l’empilement temporel forme un **spectrogramme**.

### 8.2 MFCC

Les **MFCC** (Mel-Frequency Cepstral Coefficients) projettent le spectre sur une **échelle Mel** (plus proche de la **perception humaine** des fréquences), puis appliquent un **log** et une **DCT** pour obtenir des coefficients ** décorrélés** et compacts. Ils sont **standard** en reconnaissance de la parole et en analyse des **timbres** vocaux.

On représente souvent une séquence MFCC comme une matrice \( X \in \mathbb{R}^{T \times F} \) : **T** trames temporelles, **F** coefficients par trame.

### 8.3 Normalisation par frame

Une **normalisation** (par exemple z-score ou min-max) **par trame** ou sur une fenêtre glissante améliore souvent la **robustesse** aux variations de **volume** entre candidats ou entre sessions.

### 8.4 LSTM

Un **LSTM** (Long Short-Term Memory) maintient un **état caché** \(h_t\) mis à jour à chaque pas de temps en fonction de l’entrée \(x_t\) et de l’état précédent. Des **portes** (oubli, entrée, sortie) contrôlent ce qui est **conservé** ou **oublié**. C’est adapté aux **séquences** de coefficients MFCC ou d’énergie pour capturer **prosodie** et **dynamique** sur plusieurs centaines de millisecondes.

---

## 9. Branche audio : pratique (Librosa, Sounddevice)

### 9.1 Chaîne logicielle

- **Sounddevice** : lecture **micro** en continu avec des **callbacks** ou des lectures par bloc ; réglage du **buffer** pour équilibrer **latence** et **stabilité**.  
- **Librosa** : **STFT**, **MFCC**, **RMS** (énergie), pistes pour **\(F_0\)** (via méthodes de détection de hauteur) selon la configuration du projet.

### 9.2 Paramètres projet

- **Fréquence d’échantillonnage : 16 kHz**, **mono**.  
- Extraction MFCC : ordre de grandeur **2–4 ms** par **bloc de 50 ms** sur CPU laptop récent.

### 9.3 Résultats et qualité signal

| Indicateur | Valeur |
|-----------|--------|
| Corrélation écart-type \(F_0\) (proxy stress) vs score subjectif | **0,62** sur **120** extraits annotés |
| SNR moyen après filtrage passe-bande parole | **14,3 dB** sur corpus de test interne |

Le **\(F_0\)** (fréquence fondamentale) et sa **variabilité** servent d’**indicateur** possible de tension vocale ; la corrélation **0,62** indique une relation **modérée à bonne** avec une étiquette humaine simplifiée, ce qui justifie son usage comme **une** des composantes d’un score de stress audio.

Fichier figure typique : `figures/trace_f0_intonation_librosa.png`.

---

## 10. Speech-to-text avec Whisper

### 10.1 Motivation

**Whisper** (OpenAI) est un modèle **entraîné à grande échelle** sur données multilingues et bruitées. Il offre souvent un bon **compromis** entre **qualité** (WER) et **simplicité d’intégration** par rapport à des chaînes classiques **ASR** (acoustic model + language model + lexique).

### 10.2 Usage dans le projet

- Variantes **small / medium** selon la **mémoire GPU** disponible.  
- Le texte produit alimente directement la **couche NLP** (embeddings, similarité).

### 10.3 Métriques observées (projet)

| Contexte | WER |
|----------|-----|
| Enregistrements « propres » | **6,1 %** |
| Avec bruit de type bureau | **11,4 %** |

**Latence** indicative pour un **batch** de **10 s** d’audio sur GPU : **0,8–1,4 s** selon la variante du modèle.

Fichier figure : `figures/capture_whisper_transcription_interface.png`.

---

## 11. Cognition langagière : Transformers, RoBERTa, MiniLM

### 11.1 Mécanisme d’attention

L’attention **scaled dot-product** est :

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr) V
\]

Les matrices **Q, K, V** sont des projections linéaires des entrées. L’**attention** permet de **pondérer** les positions du contexte pour chaque token. En empilant des **blocs** (multi-tête + feed-forward + normalisation), on obtient un **encodeur contextuel** puissant.

### 11.2 RoBERTa

**RoBERTa** reprend l’architecture **BERT** avec un **pré-entraînement** repensé : davantage de **données**, **pas de tâche NSP** (Next Sentence Prediction) telle que dans BERT original, autres réglages d’optimisation. En pratique : **meilleurs représentations** pour beaucoup de tâches en aval.

### 11.3 MiniLM

**MiniLM** est une approche de **distillation** permettant des modèles **plus petits** et **plus rapides**, adaptés à une **inférence CPU** fréquente dans une **UI** web.

### 11.4 Embeddings

Une phrase (ou paire de phrases) est projetée en un vecteur \(\mathbf{e} \in \mathbb{R}^d\). La **proximité** dans cet espace reflète souvent une **similarité sémantique**. On peut mesurer :

- **similarité cosinus** ;  
- **distance euclidienne** ;  
- ou combiner avec un **classifieur** si la tâche est binaire (réponse acceptable / non).

---

## 12. Scoring par similarité cosinus

### 12.1 Formule

Pour deux vecteurs \(\mathbf{a}\) (réponse attendue) et \(\mathbf{b}\) (réponse du candidat), normalisés ou non :

\[
\mathrm{sim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\lVert \mathbf{a}\rVert \,\lVert \mathbf{b}\rVert}
\]

La similarité cosinus est dans **[-1, 1]**. On peut la **mapper** vers un score **\(s \in [0,100]\)** par transformation affine ou par calibration (par exemple isotonic regression) sur un petit jeu **étiqueté** par des experts.

### 12.2 Validation par rapport à l’humain

- **Corrélation de Pearson** entre le score cosinus et une notation humaine : **0,79** sur **n = 200** paires question/réponse.  
- **Temps d’encodage** d’une paire sur **CPU** avec **MiniLM** : **8–15 ms**.

Ces ordres de grandeur montrent que le score automatique est **aligné** avec un jugement humain **dans une proportion notable de variance**, tout en restant **rapide** pour une UI interactive.

Fichier figure : `figures/histogramme_scores_similarite_cosinus.png`.

---

## 13. Cerveau décisionnel : formalisation MDP

### 13.1 États

L’état **\(s\)** est une abstraction du candidat et du contexte. Dans le projet, on l’illustre par un vecteur ou une **discrétisation** de **(Stress, Performance)** agrégés à partir de :

- sorties **vision** (niveau de stress facial ou score dérivé) ;  
- **audio** (prosodie, \(F_0\), MFCC) ;  
- **NLP** (similarité réponse, richesse lexicale éventuelle).

### 13.2 Actions

Trois familles d’actions principales :

1. **Ice-breaker** : question ou remarque pour **réduire la pression**.  
2. **Technique** : question **à contenu dur**, évaluation des compétences.  
3. **Soft skills** : **communication**, **leadership**, travail en équipe, etc.

### 13.3 Récompenses

La **récompense \(r\)** encode les **objectifs pédagogiques** :

- **Récompense positive** si la **performance** monte et/ou le **stress** baisse de façon mesurable.  
- **Pénalité** si le stress **sature** sans **gain** sur la qualité du contenu des réponses (pour éviter d’ « écraser » le candidat).

La fonction de récompense est **centrale** : une mauvaise conception peut entraîner des comportements **myopes** ou **contre-productifs** malgré un algorithme RL correct.

---

## 14. Q-learning, équation de Bellman, exploration

### 14.1 Mise à jour Q-learning (idée)

Pour une paire état-action \((s,a)\), la mise à jour tabulaire s’écrit :

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \bigr]
\]

- \(\alpha\) : **pas d’apprentissage**.  
- \(\gamma\) : **facteur d’actualisation** (importance des gains futurs).

Cette formule propage la valeur des **transitions observées** ; à la limite, sous hypothèses (exploration suffisante, disc réasonnable), \(Q\) approche la **fonction Q optimale** dans le cadre markovien.

### 14.2 Exploitation vs exploration

- **Exploitation** : choisir \(a^* = \arg\max_a Q(s,a)\).  
- **Exploration** : avec probabilité \(\varepsilon\), tirer une action **aléatoire** pour **découvrir** de meilleures stratégies.

Une **décroissance** classique :

\[
\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 \gamma_\varepsilon^t)
\]

permet beaucoup d’exploration au début et **stabilise** la politique ensuite.

Fichier figure : `figures/courbe_epsilon_decay_exploration.png`.

---

## 15. Entraînement et convergence en pratique

### 15.1 Entretiens virtuels

Le projet mentionne une **Q-table discrétisée** apprise sur **50 000 entretiens virtuels** : ce sont des **simulations** de transitions d’états avec **bruit calibré** (pour imiter la variabilité réelle) plutôt que 50 000 vrais humains — ce qui serait coûteux en données et en éthique.

### 15.2 Indicateurs de convergence

| Indicateur | Valeur |
|-----------|--------|
| Gain récompense cumulée moyenne (début → convergence) | **+38 %** (fenêtre glissante 500 épisodes) |
| Écart-type de la politique sur 10 seeds, état « stress élevé / perf moyenne » | **< 0,04** |

Ces résultats suggèrent une **convergence raisonnablement stable** sur l’espace d’états choisi, sous réserve que la **discrétisation** et la **fonction de récompense** restent représentatives des objectifs métier.

Fichier figure : `figures/convergence_recompense_qlearning_entretiens_virtuels.png`.

---

## 16. Interface web : Streamlit et machine d’états

### 16.1 Pourquoi Streamlit

**Streamlit** permet de prototyper rapidement une **UI** en Python : **widgets**, **réactivité**, et surtout **session state** pour garder des variables entre **re-exécutions** du script (modèle chargé, étape courante, buffers).

### 16.2 Machine d’états (workflow utilisateur)

États typiques du flux :

1. **ACCUEIL** — présentation, lancement session.  
2. **CALIBRAGE** — test micro/caméra, niveau sonore.  
3. **ENTRETIEN** — boucle perception + questions + scores.  
4. **DEBRIEF** — synthèse, graphiques, conseils.

Les transitions sont **explicites** dans le code pour éviter les **états incohérents** (par exemple démarrer deux fois le flux sans **reset** des buffers).

### 16.3 Bonnes pratiques UX / technique

- Pas de **double démarrage** des flux sans libération des ressources.  
- **Reset** propre des **buffers audio** entre sessions.  
- Modèles lourds chargés **une fois** et référencés via **`st.session_state`**.

Fichier figure : `figures/diagramme_state_machine_streamlit.png`.

---

## 17. Temps réel : vidéo, audio, navigateur

### 17.1 Rafraîchissement vidéo

**`st.empty()`** fournit un **emplacement** dans la page que l’on peut **réécrire** à chaque itération : idéal pour afficher une **frame OpenCV** successives sans multiplier les composants.

### 17.2 Boucle typique

1. Lire une **image** de la caméra.  
2. **Resize** pour le réseau et pour l’affichage.  
3. **Inférence** CNN (stress / expression).  
4. **Dessiner** overlays (cadre visage, score).  
5. **Publier** dans le placeholder Streamlit.

### 17.3 Audio temps réel

**Callback Sounddevice** + **file circulaire** (ring buffer) : les blocs audio entrants alimentent le calcul **MFCC** / \(F_0\) sans bloquer l’interface trop longtemps.

### 17.4 Métrique d’affichage

**15–22 FPS** stables sur laptop avec **modèle vision léger** : ce chiffre inclut **capture + inférence + rendu**, pas seulement le modèle.

Fichier figure : `figures/capture_ecran_dashboard_streamlit_temps_reel.png`.

---

## 18. Scénarios : candidat stressé vs confiant

### 18.1 Candidat stressé

- Les scores **vision + audio** indiquent une **hausse** du stress.  
- La **politique RL** favorise d’abord des actions **ice-breaker**, puis des **questions courtes** pour **reconstruire** le sentiment de maîtrise.  
- Sur un scénario type : **réduction de 12 %** du proxy **\(F_0\)** après **deux tours** (la personne se calme progressivement).

### 18.2 Candidat confiant

- Actions **techniques** **plus fréquentes** (le système « pousse » sur le fond).  
- **Similarité NLP** plus élevée : médiane **sim = 0,81** vs **0,54** pour le profil stressé sur un **banc de test** donné.

Ces comportements illustrent l’**adaptation** de la politique ; les chiffres dépendent du **calibrage** et des **jeux de test**.

Fichier figure : `figures/comparaison_scenarios_stresse_vs_confiant.png`.

---

## 19. MLOps, traçabilité et qualité

### 19.1 Versioning

- **Données** : versions de jeux annotés (vision, transcripts, scores humains).  
- **Modèles** : artefact PyTorch / ONNX + **hash** + note de commit ou identifiant DVC/MLflow.

### 19.2 Évaluation par « slices »

Performance **par segment** : niveau de **bruit**, **niveau de stress** déclaré, **langue** ou accent si applicable. Cela révèle les **failles** (Whisper faible sur un slice, CNN biaisé sur un éclairage).

### 19.3 Monitoring de dérive

Sur déploiement prolongé : **dérive des distributions** d’entrées (MFCC, luminosité caméra) et **alertes** si les scores sortent des plages historiques.

### 19.4 CI légère

**Tests unitaires** sur le pipeline audio (dimensions MFCC, pas de NaN), sur le calcul **cosinus**, sur la **machine d’états** (transitions autorisées).

---

## 20. Synthèse des métriques du projet

| Domaine | Métrique | Valeur |
|---------|----------|--------|
| Vision | Accuracy validation | **87,72 %** |
| Vision | F1 macro | **0,84** |
| Vision | Inférence | **12–18 ms**/frame (GPU) |
| Multimodal | Latence décision cible | **300–800 ms** |
| Multimodal | Sync audio-vidéo | **± 40 ms** (bench) |
| Audio | Corrélation \(F_0\) / stress annoté | **0,62** (n=120) |
| Audio | SNR après filtrage | **14,3 dB** |
| Audio | MFCC / bloc 50 ms | **2–4 ms** (CPU) |
| ASR | WER propre / bruyant | **6,1 % / 11,4 %** |
| ASR | Latence 10 s audio | **0,8–1,4 s** (GPU) |
| NLP | Pearson vs humain | **0,79** (n=200) |
| NLP | Encodage paire | **8–15 ms** (CPU) |
| RL | Entretiens simulés | **50 000** |
| RL | Gain récompense | **+38 %** |
| RL | Stabilité policy (écart-type) | **< 0,04** |
| UI | FPS affichage | **15–22** (laptop, modèle léger) |

---

## 21. Perspectives

- **Cloud** : déporter Whisper et CNN sur GPU distants pour alléger le poste client ; **autoscaling** selon la charge.  
- **Orchestration** : **Kubernetes** pour services modulaires (API vision, API ASR, service RL).  
- **LLM génératif** : générer des **feedbacks** pédagogiques **nuancés** et **personnalisés** à partir des scores multimodaux (en contrôlant **hallucinations** et **conformité** RGPD).  
- **Éthique** : consentement explicite à la **capture** biométrique, **transparence** sur l’usage des scores, **droit à l’oubli** sur les enregistrements.

---

## 22. Glossaire

| Terme | Signification courte |
|-------|----------------------|
| **ASR** | Automatic Speech Recognition (reconnaissance automatique de la parole) |
| **CNN** | Réseau convolutif pour images |
| **Embedding** | Représentation vectorielle dense d’un texte ou d’un signal |
| **FPS** | Images par seconde |
| **LSTM** | Réseau récurrent à mémoire longue |
| **MDP** | Processus de décision markovien |
| **MFCC** | Coefficients cepstraux en échelle Mel |
| **WER** | Word Error Rate (taux d’erreur sur les mots) |
| **RL** | Apprentissage par renforcement |
| **SNR** | Rapport signal sur bruit |
| **Whisper** | Modèle ASR OpenAI utilisé pour la transcription |

---

## 23. Correspondance avec les figures de la présentation

| Fichier | Rôle |
|---------|------|
| `figures/illustration_entretien_classique.png` | Illustration problématique |
| `figures/architecture_conceptuelle_coach.png` | Vue conceptuelle |
| `figures/schema_pipeline_end_to_end.png` | Pipeline |
| `figures/diagramme_fusion_multimodale.png` | Fusion |
| `figures/courbe_loss_accuracy_modele_vision.png` | Vision — entraînement |
| `figures/spectrogramme_mfcc_exemple.png` | Audio — MFCC |
| `figures/trace_f0_intonation_librosa.png` | Audio — \(F_0\) |
| `figures/capture_whisper_transcription_interface.png` | Whisper |
| `figures/histogramme_scores_similarite_cosinus.png` | NLP — scores |
| `figures/courbe_epsilon_decay_exploration.png` | RL — \(\varepsilon\) |
| `figures/convergence_recompense_qlearning_entretiens_virtuels.png` | RL — convergence |
| `figures/diagramme_state_machine_streamlit.png` | UI — états |
| `figures/capture_ecran_dashboard_streamlit_temps_reel.png` | Dashboard |
| `figures/comparaison_scenarios_stresse_vs_confiant.png` | Scénarios |

---

*Fin du document — Smart Interview Coach, description technique complète.*
