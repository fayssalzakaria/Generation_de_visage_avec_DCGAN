if __name__ == '__main__':  # Vérifie que le script est exécuté directement et non importé comme un module
    import torch.multiprocessing as mp
    # Importation des bibliothèques nécessaires
    import torch  # Framework de deep learning
    import torch.nn as nn  # Module contenant les couches de réseaux de neurones
    import torch.optim as optim  # Module pour les algorithmes d'optimisation
    import torchvision.transforms as transforms  # Module pour transformer les images
    from torch.utils.data import DataLoader  # Classe pour charger les données de manière efficace
    import os  # Module pour l'interaction avec le système de fichiers
    from model import Generator, Discriminator  # Importation des modèles de génération et de discrimination
    import matplotlib.pyplot as plt  # Module pour l'affichage des images
    import numpy as np  # Bibliothèque pour la manipulation de matrices et tableaux
    from dataset import CustomDataset  # Importation du dataset personnalisé
    import matplotlib
    mp.set_start_method('spawn', force=True)  # Force un démarrage propre des processus
    matplotlib.use('Agg')  # Désactive Tkinter

    # **Configuration des paramètres d'entraînement**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sélection du GPU si disponible, sinon CPU
    print(f"Entraînement sur : {device}")

    img_size = 64  # Taille des images (64x64 pixels)
    batch_size = 128  # Nombre d'images par batch
    epochs = 50  # Nombre total d'époques d'entraînement
    z_dim = 100  # Taille du vecteur de bruit pour le générateur (latent space)
    channels = 3  # Nombre de canaux (RGB = 3)
    feature_maps = 64  # Nombre de cartes de caractéristiques dans le réseau
    data_dir = "data/img_align_celeba/img_align_celeba"  # Chemin du dataset CelebA

    # Vérification de l'existence du dataset
    if not os.path.exists(data_dir):
        raise RuntimeError("Le dataset CelebA n'a pas été trouvé. Lance d'abord utils.py !")

    # Création d'un dossier pour stocker les images générées si non existant
    os.makedirs("generated_images", exist_ok=True)

    # **Définition des transformations à appliquer aux images**
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Redimensionnement à 64x64 pixels
        transforms.CenterCrop(img_size),  # Recadrage centré à 64x64
        transforms.ToTensor(),  # Conversion en tenseur PyTorch
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation des pixels entre [-1,1]
    ])

    # **Chargement du dataset avec le DataLoader**
    dataset = CustomDataset(root=data_dir, transform=transform)  # Création du dataset personnalisé
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)  
    # - `shuffle=True` : Mélange les images à chaque époque
    # - `num_workers=4` : Nombre de threads pour accélérer le chargement
    # - `pin_memory=True` : Optimisation pour GPU

    # **Initialisation des modèles**
    generator = Generator(z_dim, channels, feature_maps).to(device)  # Instanciation du générateur
    discriminator = Discriminator(channels, feature_maps).to(device)  # Instanciation du discriminateur

    # **Définition de la fonction de perte et des optimisateurs**
    criterion = nn.BCELoss()  # Fonction de perte pour la classification binaire (vrai/faux)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))  
    # - `lr=0.0002` : Taux d'apprentissage
    # - `betas=(0.5, 0.999)` : Paramètres de l'optimiseur Adam pour la stabilité

    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))  
    # Même configuration pour le discriminateur

    # **Fonction pour générer et sauvegarder des images à chaque époque**
    def generate_and_save_images(epoch, fixed_noise):
        """
        Génère et enregistre un lot d'images synthétiques à partir d'un bruit fixé.
        :param epoch: Numéro de l'époque en cours.
        :param fixed_noise: Vecteur de bruit constant pour suivre l'évolution du modèle.
        """
        generator.eval()  # Mode évaluation (désactive le dropout et la BN)
        with torch.no_grad():
            fake_images = generator(fixed_noise).cpu()  # Génération des images

        fake_images = (fake_images + 1) / 2  # Re-normalisation des images vers [0,1]

        # Affichage des images générées sous forme de grille 4x4
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            img = np.transpose(fake_images[i].numpy(), (1, 2, 0))  # Réorganisation des dimensions (H, W, C)
            ax.imshow(img)  # Affichage de l'image
            ax.axis("off")  # Suppression des axes
        
        # Sauvegarde de l'image
        plt.savefig(f"generated_images/epoch_{epoch}.png")
        plt.close(fig)
        plt.close()
        generator.train()  # Retour en mode entraînement

    # **Création d'un vecteur de bruit fixe pour suivre la progression du modèle**
    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)  

    # **Boucle d'entraînement du GAN**
    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):  
            # Récupération d'un batch d'images réelles
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # **Création des labels pour l'entraînement**
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Labels réels légèrement inférieurs à 1 (smoothing)
            fake_labels = torch.zeros(batch_size, 1, device=device)  # Labels pour les images générées (0)

            # **Entraînement du discriminateur**
            optimizer_D.zero_grad()  # Réinitialisation du gradient
            output_real = discriminator(real_images)  # Prédiction sur les images réelles
            loss_real = criterion(output_real, real_labels)  # Calcul de la perte pour les réelles

            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)  # Génération du bruit
            fake_images = generator(noise)  # Création des images synthétiques
            output_fake = discriminator(fake_images.detach())  # Prédiction sur les fausses images
            loss_fake = criterion(output_fake, fake_labels)  # Calcul de la perte pour les fausses images

            loss_D = loss_real + loss_fake  # Somme des pertes
            loss_D.backward()  # Rétropropagation
            optimizer_D.step()  # Mise à jour des poids du discriminateur

            # **Entraînement du générateur**
            optimizer_G.zero_grad()  # Réinitialisation du gradient
            output_fake = discriminator(fake_images)  # Nouvelle prédiction sur les fausses images
            loss_G = criterion(output_fake, real_labels)  # Le générateur veut tromper le discriminateur
            loss_G.backward()  # Rétropropagation
            optimizer_G.step()  # Mise à jour des poids du générateur

            # Affichage des pertes toutes les 100 itérations
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}] - "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        # Sauvegarde des images générées à la fin de chaque époque
        generate_and_save_images(epoch+1, fixed_noise)

    print("Entraînement terminé !")  # Indique la fin de l'entraînement
