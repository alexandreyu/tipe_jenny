import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(images,target),(  i ,  m ) = fashion_mnist.load_data()
images= images[:10000]
targets= target[:10000]

print(images.shape,targets.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(images[10])
plt.title(class_names[targets[11]])
plt.show()
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()



train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure()
num_fring = int(rd.random()*60000)
print(class_names[train_labels[num_fring]])
plt.imshow(train_images[num_fring])
plt.colorbar()
plt.grid(False)
plt.show()
'''