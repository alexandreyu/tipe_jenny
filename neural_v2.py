import tensorflow as tf
import matplotlib.pyplot as plt


# La taille de paquet est le nombre d'images par paquet, ça permet de paralléliser et de traiter beaucoup plus vite
batch_size = 8
size = (120, 160)

# On charge les images dans nos deux datasets
(train_set, test_set) = tf.keras.utils.image_dataset_from_directory("dataset",
                                                                    subset="both",
                                                                    validation_split=0.2,
                                                                    seed=4269,
                                                                    image_size=size,
                                                                    batch_size=batch_size)
# On définit les catégories (gauche, milieu, droite)
class_names = train_set.class_names

# On charge les images en mémoire plutôt que aller les chercher dans les datasets à chaque fois
AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

n_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(size[0], size[1], 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Affiche la structure des couches, plus pour le style qu'autre chose
model.summary()

epochs = 100
epochs_range = range(epochs)

history = model.fit(
  train_set,
  validation_data=test_set,
  epochs=epochs
)

test_loss, test_acc = model.evaluate(test_set, verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
