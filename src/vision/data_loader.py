import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir="data/", batch_size=32):
    """
    Cette fonction prépare les images et crée les DataLoaders pour l'entraînement.
    """
    
    # 1. Définir les transformations mathématiques et visuelles
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # On s'assure que c'est en noir et blanc (1 canal)
        transforms.Resize((48, 48)),                 # On force la taille à 48x48 pixels
        transforms.ToTensor(),                       # On convertit l'image en Matrice PyTorch (Tenseur)
        transforms.Normalize(mean=[0.5], std=[0.5])  # On normalise les valeurs pour faciliter les calculs
    ])

    # 2. Indiquer où se trouvent les dossiers d'entraînement et de test
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 3. Créer les Datasets (PyTorch va lire les dossiers et associer les étiquettes automatiquement)
    # Note : ImageFolder s'attend à ce que chaque sous-dossier soit le nom d'une émotion
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # 4. Créer les DataLoaders (les serveurs de cantine qui font des paquets de 'batch_size')
    # shuffle=True mélange les images d'entraînement pour que le modèle n'apprenne pas par cœur l'ordre
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

# Petit test pour s'assurer que le script fonctionne (même sans les images pour l'instant)
if __name__ == "__main__":
    print("Le script data_loader.py est prêt ! 🚀")
    print("Il attend que les images soient placées dans le dossier 'data/train' et 'data/test'.")