import tensorflow as tf
'''
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import PIL
'''
# On charge les données du dataset dans leurs ensembles respectifs
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

'''
Train_images contient les images qui serviront à entrainer le modèle
Train_labels contient les "étiquettes" des images d'entrainement, la catégorie associée à chaque image
Test_images contient les images qui serviront à évaluer les performances du modèle
Test_labels contient les étiquettes des images de test
'''

# On définit les catégories des images, en l'occurence les types de vêtements
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# On normalise les images (couleur entre 0 et 1 au lieu de 0 et 255)
train_images = train_images / 255.0
test_images = test_images / 255.0

# On crée le modèle
# Sequential crée un réseau séquentiel avec les couches qui lui sont fournies
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten permet de créer une couche avec un neurone pour chaque
    # pixel de l'image, soit ici 28x28 neurones
    tf.keras.layers.Dense(128, activation='relu'),  # Dense est une couche dont tous les neurones sont reliés
    # à tous les neurones précédents ; relu est une fonction d'activation telle que f(x) = max(0, x)
    tf.keras.layers.Dense(10)  # Un neurone par catégorie de vêtement
])

'''
On a donc un modèle qui contient 3 couches. La première contient 784 neurones (1 par pixel) et qui prend en entrée la
couleur (en noir et blanc) du pixel. La deuxième contient 128 neurones tous reliés à tous les précédents et qui 
renvoient 0 si leur résultat est négatif, et la valeur de leur résultat sinon. La troisième contient 10 neurones 
(1 par catégorie de vêtement), et ils renvoient le degré de certitude avec lequel le modèle classe le vêtement (ex, si
le neurone qui représente les tshirts renvoie 0,99 celui qui représente les robes 0,01 (et que tous les autres neurones
sont négligeables), alors le modèle renverra tshirt 99% du temps et robe 1% du temps pour cette image).
'''

# On compile le modèle en choisissant l'algorithme, la fonction de perte et les mesures
model.compile(optimizer='adam',  # Algorithme de descente de gradient
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Fonction de perte, ça fait des
              # trucs compliqués et c'est pas très important de savoir comme ça marche
              metrics=['accuracy'])  # On mesure la précision, la fraction d'images correctement catégorisées

# Une fois que le modèle est compilé, on l'entraine :
model.fit(train_images, train_labels, epochs=10)  # Epochs est le nombre de sessions d'entrainement

# Et ensuite, on lui fait passer son évaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nPrécision test : ', test_acc)
print('\nPerte test : ', test_loss)

'''
On peut voir que la précision atteinte lors de l'entrainement est proche de 91%, alors que celle obtenue lors de 
l'évaluatino est de 87%, ce qui veut dire que le modèle est légèrement overfit (il apprend par coeur plutot que de 
reconnaitre des motifs). Ce problème s'aggrave si on augmente beaucoup le nombre d'epochs (le modèle finit par connaitre
toutes les images d'entrainement par coeur mais pas celles de test), mais peut être mitigé en augmentant la taille et la
diversité des images d'entrainement, ou en changeant la structure du réseau de neurones. Choisir le nombre, la taille et
le type des couches cachées entre la couche d'entrée et la couche de sortie relève bien plus de l'art que de la science,
il est possible d'expérimenter complètement au pif, ou d'avoir des intuitions basées sur des expériences précédentes.
'''

'''
J'ai testé avec 100 epochs, on atteint 98.04% de précision en entrainement (très bon !), mais seulement 88.20% lors de
l'évaluation, ce qui veut dire que le modèle apprend par coeur les images d'entrainement et sous-performe face à des 
images inconnues, ce qui n'est pas bon.
'''