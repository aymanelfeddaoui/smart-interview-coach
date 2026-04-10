import torch
import torch.nn as nn
import torch.optim as optim

from src.vision.data_loader import get_data_loaders
from src.vision.emotion_model import EmotionCNN
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def train_deep_model(num_epochs=30):
    print("1. Préparation des données...")
    train_loader, test_loader, classes = get_data_loaders()

    print("2. Initialisation du modèle ResNet...")
    model = EmotionCNN(num_classes=7)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # La variable qui va mémoriser notre record absolu (initialisée à l'infini)
    best_val_loss = float('inf') 

    print(f"\n3. Lancement de l'entraînement profond sur {num_epochs} époques...")
    print("   (Laisse tourner ce script, cela peut prendre du temps !)\n")
    
    for epoch in range(num_epochs):
        # --- PHASE 1 : APPRENTISSAGE (TRAIN) ---
        model.train()
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)

        # --- PHASE 2 : L'EXAMEN BLANC (VALIDATION) ---
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                # On calcule aussi le pourcentage de réussite pour voir si l'IA s'améliore
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        avg_val_loss = running_val_loss / len(test_loader)
        accuracy = 100 * correct_predictions / total_predictions

        print(f"Époque [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Précision: {accuracy:.2f}%")

        # --- PHASE 3 : SAUVEGARDE INTELLIGENTE ---
        # Si l'erreur à l'examen est la plus basse jamais vue, on sauvegarde !
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # On le nomme "best" pour ne pas écraser ton brouillon "v1"
            torch.save(model.state_dict(), 'models/emotion_model_best.pth')
            print("   🌟 Nouveau record ! Modèle sauvegardé.")

    print("\nEntraînement profond terminé ! 🧠")
    print("Le meilleur cerveau a été sauvegardé dans 'models/emotion_model_best.pth'")

if __name__ == "__main__":
    # Tu pourras ajuster num_epochs plus tard, on commence avec 30
    train_deep_model(num_epochs=30)