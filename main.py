import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
import os 

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
import os

def creation_image_balle_noir(i,j,r,L,l):
    Im=[]
    for k in range(L):
        M=[]
        for m in range (l):
            if ((i-k)**2+(j-m)**2)<= r**2:
                M.append( [247,134,1])
            else:
                M.append([0,0,0])
        Im.append(M)
    return np.array(Im, dtype=np.uint8)

def point_aleatoire(r,L,l):
    i= random.randint(r,L-r)
    j= random.randint(r,l-r)
    return creation_image_balle_noir(i,j,r,L,l)

def creation_basedonnee(N):
    for i in range (N):
        image = point_aleatoire(3,100,150)
        plt.imshow(image)
        plt.imsave(r"C:\Users\jxue0\OneDrive\Documents\Jenny\Doc prépa\TIPE LAKANAL\bade de donnee\data"+str(i)+".png",image)
        print("Les fichiers seront enregistrés dans :", os.getcwd())


plt.show()
creation_basedonnee(3)
