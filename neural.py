import tensorflow as tf
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import PIL

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure()
num_fring = int(rd.random()*60000)
print(class_names[train_labels[num_fring]])
plt.imshow(train_images[num_fring])
plt.colorbar()
plt.grid(False)
plt.show()
