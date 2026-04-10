import cv2
import torch
import numpy as np
from torchvision import transforms
from src.vision.emotion_model import EmotionCNN

def analyser_stress_webcam(duree_secondes=5):
    """
    Allume la webcam pendant quelques secondes, analyse les émotions,
    et renvoie un score de stress (0.0 à 1.0) basé sur la moyenne.
    """
    print("1. Chargement du modèle visuel (Le meilleur cerveau)...")
    # On charge l'architecture
    model = EmotionCNN(num_classes=7)
    # On charge les poids que tu as entraînés cette nuit !
    model.load_state_dict(torch.load('models/emotion_model_best.pth', weights_only=True))
    model.eval() # Mode inférence (pas d'entraînement)

    # L'outil d'OpenCV pour trouver les visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Dictionnaire des émotions (les mêmes que dans l'entraînement)
    emotions = ['colere', 'degout', 'peur', 'joie', 'neutre', 'tristesse', 'surprise']
    
    # Les transformations nécessaires pour que l'image ressemble à celles de l'entraînement
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("2. Allumage de la webcam...")
    cap = cv2.VideoCapture(0) # 0 correspond généralement à la webcam principale
    
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam.")
        return 0.5 # Stress neutre par défaut si pas de caméra

    scores_stress = []
    
    # On utilise OpenCV pour récupérer le temps exact
    start_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    print(f"3. Analyse en direct pendant {duree_secondes} secondes...")
    
    while True:
        # Lire l'image actuelle
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir en noir et blanc pour le détecteur de visage
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Trouver les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Découper le visage
            roi_gray = gray[y:y+h, x:x+w]
            
            # Appliquer les transformations PyTorch
            tensor_visage = transform(roi_gray).unsqueeze(0)
            
            # Demander à l'IA
            with torch.no_grad():
                outputs = model(tensor_visage)
                _, predicted = torch.max(outputs, 1)
                emotion_predite = emotions[predicted.item()]
                
            # Calcul du stress
            if emotion_predite in ['peur', 'colere', 'tristesse']:
                scores_stress.append(0.8)
            elif emotion_predite in ['joie', 'neutre']:
                scores_stress.append(0.2)
            else:
                scores_stress.append(0.5)

            # Dessiner un rectangle rouge autour du visage pour le style
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion_predite, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Afficher la fenêtre vidéo
        cv2.imshow('Analyse du Candidat (Appuyez sur Q pour quitter)', frame)

        # Vérifier le temps écoulé
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - start_time) / freq
        
        if time_elapsed > duree_secondes:
            break
            
        # Si on appuie sur la touche 'q', on arrête avant la fin
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    
    print("4. Fin de la capture.")
    
    # On renvoie la moyenne de stress calculée
    if len(scores_stress) > 0:
        stress_moyen = sum(scores_stress) / len(scores_stress)
        return stress_moyen
    else:
        return 0.5 # Si aucun visage n'a été trouvé

if __name__ == "__main__":
    stress = analyser_stress_webcam(duree_secondes=3)
    print(f"\nScore de stress calculé par l'IA : {stress:.2f} (0.0 = Zen, 1.0 = Panique)")