import torch

path = r"C:\Users\hp\Desktop\smart_interview_coach\models\audio_model_best.pth"

data = torch.load(path, map_location=torch.device('cpu'))

print("\n📦 CONTENU COMPLET :")
print(data)
if isinstance(data, dict):
    print("\n🔍 Recherche des métriques :")

    for key in data.keys():
        if "loss" in key.lower() or "acc" in key.lower():
            print(f"✅ Trouvé : {key} = {data[key]}")
print("Accuracy :", data['accuracy'])
print("Loss :", data['loss'])