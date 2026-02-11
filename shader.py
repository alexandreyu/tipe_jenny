import cv2 as cv
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt


def convert_to_png(path):
    img = Image.open(path + ".jpg")
    img.save(path + ".png")


def convert_folder(path):
    for i in os.listdir(path):
        convert_to_png(path + "/" + i.split(".")[0])

'''
Shader (ici) : instruction qui est réalisée pour chaque pixel d'une image (ou d'un rendu 3D), sert à faire des 
effets de lumière, ombres, rendus particuliers...

Principe de la détection par shader : on parcourt tous les pixels de l'image et on vérifie si sa teinte (hue) est 
comprise entre certaines valeurs (qui sont essentiellement entrées au pif jusqu'à que ça filtre bien le fond sans être
trop restrictif), et on les colorie en rouge.

À implémenter : algorithme qui prend en entrée l'image avec la boule en rouge, et qui renvoie le centre
Possible de combiner les deux algo pour ne pas passer par l'étape de coloriage, mais complexe en temps réel peut-être.



'''


def draw_crosshair(image, x, y):
    image[x, y] = [0, 0, 0]
    image[x + 1, y] = [0, 0, 0]
    image[x + 2, y] = [0, 0, 0]
    image[x - 1, y] = [0, 0, 0]
    image[x - 2, y] = [0, 0, 0]
    image[x, y + 1] = [0, 0, 0]
    image[x, y + 2] = [0, 0, 0]
    image[x, y - 1] = [0, 0, 0]
    image[x, y - 2] = [0, 0, 0]
    return image


def process_image(image):
    center = []
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            strongest = max(int(pixel[0]), int(pixel[1]), int(pixel[2]))
            fetch_condition = (
                    (pixel[2] / strongest >= 0.9) and (pixel[2] / strongest <= 1.0) and  # Rouge
                    (pixel[1] / strongest >= 0.7) and (pixel[1] / strongest <= 1.0) and  # Vert
                    (pixel[0] / strongest >= 0.5) and (pixel[0] / strongest <= 0.8)  # Bleu
            )
            if fetch_condition:
                count += 1
                if len(center) == 2:
                    center[0] += i
                    center[1] += j
                else:
                    center.append(i)
                    center.append(j)
    if len(center) != 2:
        center.append(0)
        center.append(0)
        count = 1

    center[0] /= count
    center[1] /= count
    return center


def crosshair_image(image, outpath="test.png"):
    center = process_image(image)
    cv.imwrite(outpath, draw_crosshair(image, int(center[0]), int(center[1])))


def crosshair_folder(path, outpath):
    for i in os.listdir(path):
        img = cv.imread(path + "/" + i)
        crosshair_image(img, outpath + "/" + i)


# print(os.listdir("shader_test"))
# convert_folder("shader_test")
crosshair_folder("shader_test", "shader_processed")
