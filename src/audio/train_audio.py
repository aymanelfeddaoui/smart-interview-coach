import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim

from src.audio.audio_loader import get_audio_loaders
from src.audio.audio_model import AudioLSTM

def train_deep_audio_model(num_epochs=50):
    print("1. Préparation de la cantine audio...")
    train_loader, test_loader, classes = get_audio_loaders()

    print("2. Création du cerveau LSTM...")
    model = AudioLSTM(num_classes=7)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    print(f"\n3. Lancement de l'entraînement profond sur {num_epochs} époques...")
    
    for epoch in range(num_epochs):
        # --- PHASE 1 : ENTRAÎNEMENT ---
        model.train()
        running_train_loss = 0.0
        
        for mfcc_tensors, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(mfcc_tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)

        # --- PHASE 2 : EXAMEN BLANC (VALIDATION) ---
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mfcc_tensors, labels in test_loader:
                outputs = model(mfcc_tensors)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = running_val_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f"Époque [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Précision: {accuracy:.2f}%")

        # --- PHASE 3 : SAUVEGARDE INTELLIGENTE ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/audio_model_best.pth')
            print("   🌟 Nouveau record ! Modèle audio sauvegardé.")

    print("\nEntraînement audio profond terminé ! 🎧")

if __name__ == "__main__":
    train_deep_audio_model(num_epochs=100)

