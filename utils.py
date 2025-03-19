import os
import kaggle
import zipfile

# Création du dossier
os.makedirs("data", exist_ok=True)

# Téléchargement depuis Kaggle
print("Téléchargement de CelebA depuis Kaggle...")
kaggle.api.dataset_download_files("jessicali9530/celeba-dataset", path="data/", unzip=True)

# Extraction des images
zip_path = "data/img_align_celeba.zip"
if os.path.exists(zip_path):
    print("Extraction des images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/")
    os.remove(zip_path)

print("Téléchargement et extraction terminés.")
