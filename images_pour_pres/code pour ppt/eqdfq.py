import matplotlib.pyplot as plt
import numpy as np
import random

def creation_image_balle(i,j,r,L,l):
    Im=[]
    for k in range(L):
        M=[]
        for m in range (l):
            if ((i-k)**2+(j-m)**2)<= r**2:
                M.append( [247,134,1])
            else:
                M.append([255,255,255])
        Im.append(M)
    return np.array(Im, dtype=np.uint8)

def point_aleatoire(r,L,l):
    i= random.randint(r,L-r)
    j= random.randint(r,l-r)
    return creation_image_balle(i,j,r,L,l),i,j

def creation_basedonnee(N):
    for i in range (N):
        image,y,x = point_aleatoire(3,120,160)
        plt.imsave(chemin+str(x)+"_"+ str(i)+".png",image)



creation_basedonnee(5000)



def trier_images_par_x_direct(dossier_source, dossier_destination):
    src_dir = Path(dossier_source)
    dest_dir = Path(dossier_destination)


    for fichier in src_dir.glob("*.png"):
        nom_fichier = fichier.name  # Exemple: "98_14.png"

        try:
            # 3. Extraction directe de x (on prend tout ce qui est avant le premier '_')
            partie_x = nom_fichier.split("_")[0]
            valeur_x = int(partie_x)

            # 4. RÈGLES DE TRI
            if valeur_x <= 32:
                dossier_cible = categories["gauche_extr"]
            elif 32< valeur_x <= 64:
                dossier_cible = categories["gauche"]
            elif 64< valeur_x <=96:
                dossier_cible = categories["milieu"]
            elif 96< valeur_x <=128:
                dossier_cible = categories["droite"]
            else:  # supérieur à 80
                dossier_cible = categories["droite_extr"]

            # 5. Déplacement physique du fichier
            shutil.move(str(fichier), str(dossier_cible / nom_fichier))
            print(f"Déplacé : {nom_fichier} -> {dossier_cible.name}")

        except (IndexError, ValueError):
            # Si un fichier ne commence pas par un nombre (ex: "icone.png"), on l'ignore proprement
            print(f"Fichier ignoré (format incorrect) : {nom_fichier}")


# --- ZONE DE LANCEMENT ---
dossier_origine = (
    "C:/Users/jxue0/Desktop/tipe/tipe_jenny/dataset_num2/base de donnees"

)
dossier_trié = "C:/Users/jxue0/Desktop/tipe/tipe_jenny/dataset_num2"

trier_images_par_x_direct(dossier_origine, dossier_trié)