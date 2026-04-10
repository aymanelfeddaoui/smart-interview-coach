import cv2
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Importer notre architecture de modèle
from src.vision.emotion_model import EmotionCNN

# 1. Configuration initiale
# L'ordre alphabétique exact généré par PyTorch lors de l'entraînement
CLASSES = ['confiance', 'degout', 'joie', 'peur', 'stress', 'surprise', 'tristesse']

print("Chargement du modèle d'Intelligence Artificielle...")
# On crée une coquille vide avec 7 classes
model = EmotionCNN(num_classes=7)
# On charge les "connaissances" (les poids) que nous avons entraînées
model.load_state_dict(torch.load('models/emotion_model_v1.pth'))
# On met le modèle en mode "évaluation" (il n'apprend plus, il devine juste)
model.eval()

# Les mêmes transformations qu'à l'entraînement
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Charger le détecteur de visage OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Allumer la caméra
cap = cv2.VideoCapture(0)
print("Caméra activée. Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en noir et blanc pour la détection du visage
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Dessiner le rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 3. Isoler le visage (Crop)
        # On découpe la zone du visage dans l'image en niveaux de gris
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Convertir en image "PIL" (le format attendu par nos transformations PyTorch)
        roi_pil = Image.fromarray(roi_gray)
        
        # Appliquer les transformations (48x48, Tenseur) et ajouter une dimension pour le "batch"
        # Le modèle attend un paquet (batch), donc on transforme [1, 48, 48] en [1, 1, 48, 48]
        tensor_image = transform(roi_pil).unsqueeze(0)

        # 4. Prédire l'émotion
        with torch.no_grad(): # On dit à PyTorch de ne pas calculer de gradients (on n'apprend pas ici)
            outputs = model(tensor_image)
            # On prend la classe qui a le score le plus élevé
            _, predicted = torch.max(outputs, 1)
            emotion_idx = predicted.item()
            emotion_label = CLASSES[emotion_idx]

        # 5. Afficher le texte sur la vidéo
        cv2.putText(frame, emotion_label.upper(), (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Afficher le résultat final
    cv2.imshow('Smart Interview Coach - Live Emotion', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()