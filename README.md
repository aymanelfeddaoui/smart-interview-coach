# 🤖 Smart Interview Coach

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)

Un simulateur d'entretien d'embauche autonome et intelligent, développé par Imad Bouferdou, Ayman El Feddaoui et Zakaria Mokarram (ENSIAS). 

Ce système analyse en temps réel les compétences et l'état émotionnel du candidat pour adapter dynamiquement la stratégie d'entretien.

## 🌟 Fonctionnalités (Pipeline End-to-End)
* **Perception Visuelle (Vision) :** Détection du niveau de stress via une analyse faciale en direct (CNN / ResNet + OpenCV).
* **Perception Audio :** Extraction de l'intonation (MFCC) et transcription vocale robuste (Whisper OpenAI).
* **Cognition Sémantique (NLP) :** Évaluation de la pertinence des réponses via Similarité Cosinus (all-MiniLM-L6-v2).
* **Cerveau Stratégique (RL) :** Prise de décision adaptative pilotée par un agent Q-Learning entraîné sur 50 000 simulations.

## 🚀 Installation & Utilisation

1. Clonez ce dépôt :
`git clone https://github.com/VOTRE_NOM/smart-interview-coach.git`

2. Installez les dépendances :
`pip install -r requirements.txt`

3. Lancez l'interface web :
`streamlit run app.py`

## 🏗️ Architecture
Le système repose sur une Machine à États (State Machine) dans Streamlit, permettant une capture multimodale (Webcam + Micro) non-bloquante directement dans le navigateur.