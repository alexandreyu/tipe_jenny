from PIL import Image
import tensorflow as tf
import os
import shutil
# Résolution : 160 x 120


def grayscale(path, gs=False):
    # Passe l'image path en noir et blanc, si gs = True rajoute gs devant le nom
    img = Image.open(path)
    out = img.convert("L")
    outpath = ""
    if gs:
        outpath += "gs_"
    outpath += path
    out.save(outpath)
    return outpath


def grayscale_folder(path):
    # Passe en noir et blanc toutes les images dans le dossier path
    files = os.listdir(path)
    for i in files:
        grayscale(path + "/" + i)


def grayscale_to_folder(path, outpath):
    # Passe en noir et blanc toutes les images dans le dossier path vers le dossier outpath
    files = os.listdir(path)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in files:
        shutil.copy2(path + "/" + i, outpath)
        grayscale(outpath + "/" + i)


def load_png(path, channels=1):
    # Charge le png path dans le retour img, channels est le nombre de couleurs (noir et blanc de base)
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=channels)
    return img


def grayscale_to_dataset():
    for i in os.listdir("images"):
        grayscale_to_folder("images/" + i, "dataset/" + i)




# grayscale_to_dataset()
# print(load_png(grayscale("cat_test.png", gs=True)))
