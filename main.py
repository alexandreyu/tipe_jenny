import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
import os 
import PIL
import serial

def creation_image_balle_noir(i,j,r,R):
    Im=[]
    for k in range(R):
        L=[]
        for l in range (R):
            if ((i-k)**2+(j-l)**2)<= r**2:
                L.append( [1,1,1,1])
            else:
                L.append([0,0,0,1])
        Im.append(L)
    return np.array(Im)

def send_to_arduino(data:list, com="COM3"):
    port = serial.Serial(com, 9600)
    port.write(data.encode('utf- d8'))

def __main__():
    print("testtetest")
    # test = np.ones((20, 20))
    S = creation_image_balle_noir(5,6,1,10)


    data = np.zeros((512, 512, 3), dtype=np.uint8) * 255
    data[:, :] = [255, 255, 255]

    plt.imshow(S)
    plt.show(data)
    # S2 = np.array(S,dtype=np.float32)

__main__()



