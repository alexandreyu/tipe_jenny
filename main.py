import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
import os 

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
import serial

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
    return np.array(Im)

def point_aleatoire(r,L,l):
    i= random.randint(0,L-r)
    j= random.randint(0,l-r)
    plt.imshow(creation_image_balle_noir(i,j,r,L,l))
    return creation_image_balle_noir(i,j,r,L,l)


print(point_aleatoire(3,100,200))
plt.show()
