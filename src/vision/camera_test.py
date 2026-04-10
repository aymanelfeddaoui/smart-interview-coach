import cv2

# ÉTAPE 1 : Charger le "détecteur de visages" pré-entraîné d'OpenCV
# OpenCV possède des modèles tout prêts pour détecter des objets basiques.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ÉTAPE 2 : Initialiser la caméra
cap = cv2.VideoCapture(0)
print("Recherche de visages en cours... Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()

    if ret == True:
        # NOUVEAU : Convertir l'image en noir et blanc (niveaux de gris)
        # Les algorithmes de détection sont beaucoup plus rapides sur des images en noir et blanc 
        # car la matrice de données est plus petite (1 canal de couleur au lieu de 3).
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # NOUVEAU : Détecter les visages dans l'image en noir et blanc
        # Cela nous renvoie une liste de coordonnées (x, y, largeur, hauteur) pour chaque visage trouvé.
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # NOUVEAU : Dessiner un rectangle autour de chaque visage trouvé
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, point_haut_gauche, point_bas_droit, couleur_BGR, epaisseur_ligne)
            # Ici on dessine un rectangle vert (0, 255, 0) d'épaisseur 2 sur l'image originale en couleur.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Afficher l'image finale avec les rectangles
        cv2.imshow('Smart Interview Coach - Detection Visage', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()