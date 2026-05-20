import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 1
size = (120, 160)

(train_set, test_set) = tf.keras.utils.image_dataset_from_directory("dataset_num2",
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

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(size[0], size[1], 3)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),padding="same",  activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes),
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
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Précision Entrainement')
plt.plot(epochs_range, val_acc, label='Précision Evaluation')
plt.legend(loc='lower right')
plt.title('Evolution de la précision du réseau')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perte Entrainement')
plt.plot(epochs_range, val_loss, label='Perte Evaluation')
plt.legend(loc='upper right')
plt.title("Evolution de l'erreur" )
plt.show()
