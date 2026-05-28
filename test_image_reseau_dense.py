
import tensorflow as tf
import matplotlib.pyplot as plt


# La taille de paquet est le nombre d'images par paquet, ça permet de paralléliser et de traiter beaucoup plus vite
batch_size = 1
size = (120, 160)

# On charge les images dans nos deux datasets
(train_set, test_set) = tf.keras.utils.image_dataset_from_directory(r"C:\Users\jxue0\Desktop\tipe\tipe_jenny\dataset",
                                                                    subset="both", #renvoie les 2 sous-ensembles
                                                                    validation_split=0.2, #20%test
                                                                    seed=1,
                                                                    image_size=size,
                                                                    batch_size=batch_size)

# On définit les catégories (gauche, milieu, droite)
class_names = train_set.class_names

# On charge les images en mémoire plutôt que aller les chercher dans les datasets à chaque fois
AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

#cache pour optimiser le temps + accès rapide après la première époque
#shuffle pour mélanger les données
#prefetch prépare des images à envoyer dès qu'une image a été traitée à la vitesse autotune
            #qui considère la capacité de l'ordi pour stocket autant que possible (buffer c'est le réservoir)

n_classes = len(class_names)


#réseau dense

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(size[0], size[1], 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_classes) #on a 3 neurones en sortie
])



model.compile(optimizer='adam', #modifie les paramètres pendant l'entraînement
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                #fonction de perte, à partir de données brut en sortir de la dernière couche
              metrics=['accuracy'])
# Affiche la structure des couches, plus pour le style qu'autre chose
model.summary()

epochs = 20
epochs_range = range(epochs)

historique = model.fit(
  train_set,
  validation_data=test_set,
  epochs=epochs
)

#on teste l'entraînement
test_loss, test_acc = model.evaluate(test_set, verbose=2)

acc = historique.history['accuracy']
val_acc = historique.history['val_accuracy']

loss = historique.history['loss']
val_loss = historique.history['val_loss']

plt.figure(figsize=(8, 8))

plt.plot(epochs_range, acc, label='Précision Entrainement')
plt.plot(epochs_range, val_acc, label='Précision Evaluation')
plt.xlabel("Epochs")
plt.ylabel("Précision")
plt.legend(loc='lower right')
plt.title('Evolution de la précision du réseau')
plt.show()

'''*
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perte Entrainement')
plt.plot(epochs_range, val_loss, label='Perte Evaluation')
plt.legend(loc='upper right')
plt.title("Evolution de l'erreur" )
plt.show()
'''


