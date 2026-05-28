import os
import shutil
import cv2

# ==========================================
# CONFIGURATION DES DOSSIERS (À MODIFIER)
# ==========================================
DOSSIER_SOURCE = "Dataset"

# Dossiers de destination selon la touche
DOSSIERS_DESTINATION = {
    "gauche": "gauche",
    "droite": "droite",
    "milieu": "haut",
    "poubelle": "bas",
}

# Extensions d'images acceptées
EXTENSIONS_VALIDES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def initialiser_dossiers():
    """Crée les dossiers de destination s'ils n'existent pas."""
    for dossier in DOSSIERS_DESTINATION.values():
        if not os.path.exists(dossier):
            os.makedirs(dossier)
            print(f"Création du dossier : {dossier}")


def trier_images():
    initialiser_dossiers()

    # Récupérer la liste de toutes les images du dossier source
    if not os.path.exists(DOSSIER_SOURCE):
        print(
            f"Erreur : Le dossier source '{DOSSIER_SOURCE}' n'existe pas. Créez-le ou modifiez le chemin."
        )
        return

    images = [
        f
        for f in os.listdir(DOSSIER_SOURCE)
        if f.lower().endswith(EXTENSIONS_VALIDES)
    ]

    if not images:
        print("Aucune image à trier dans le dossier source !")
        return

    print(f"--- Début du tri : {len(images)} images trouvées ---")
    print("Commandes : Flèches (Gauche/Droite/Haut/Bas) pour trier.")
    print("Pressez 'Échap' (ESC) ou 'q' pour quitter.\n")

    # Créer une fenêtre OpenCV redimensionnable
    cv2.namedWindow("Trieur d'images", cv2.WINDOW_NORMAL)

    for i, nom_image in enumerate(images):
        chemin_complet_source = os.path.join(DOSSIER_SOURCE, nom_image)

        # Charger et afficher l'image
        img = cv2.imread(chemin_complet_source)
        if img is None:
            print(f"Impossible de lire l'image : {nom_image}, passage à la suivante.")
            continue

        cv2.imshow("Trieur d'images", img)
        print(f"[{i+1}/{len(images)}] Affichage de : {nom_image}")

        while True:
            # waitKey(0) attend qu'une touche soit pressée (0 = indéfiniment)
            touche = cv2.waitKeyEx(0)

            dossier_cible = None

            # Détection des touches fléchées (les codes peuvent varier selon l'OS)
            # OpenCV utilise des codes spécifiques pour les flèches avec waitKeyEx
            if touche in (2424832, 65361, 81):  # Codes flèche gauche Windows/Linux/Mac
                dossier_cible = DOSSIERS_DESTINATION["gauche"]
                direction = "GAUCHE"
            elif touche in (2555904, 65363, 83):  # Flèche droite
                dossier_cible = DOSSIERS_DESTINATION["droite"]
                direction = "DROITE"
            elif touche in (2490368, 65362, 82):  # Flèche haut
                dossier_cible = DOSSIERS_DESTINATION["milieu"]
                direction = "HAUT"
            elif touche in (2621440, 65364, 84):  # Flèche bas
                dossier_cible = DOSSIERS_DESTINATION["poubelle"]
                direction = "BAS"

            # Touches pour quitter : Échap (27) ou 'q' (113)
            elif touche in (27, 113, ord("q"), ord("Q")):
                print("\nArrêt du tri par l'utilisateur.")
                cv2.destroyAllWindows()
                return

            # Si une direction valide a été choisie
            if dossier_cible:
                chemin_complet_dest = os.path.join(dossier_cible, nom_image)
                try:
                    # Déplace l'image vers le dossier cible
                    shutil.move(chemin_complet_source, chemin_complet_dest)
                    print(f" -> Déplacée vers {direction}\n")
                except Exception as e:
                    print(f"Erreur lors du déplacement de l'image : {e}")
                break  # Sort de la boucle d'attente pour passer à l'image suivante

    print("Félicitations, toutes les images ont été triées !")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    trier_images()