import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 1
size = (120, 160)

(train_set, test_set) = tf.keras.utils.image_dataset_from_directory("dataset_numérique",
                                                                    subset="both",
                                                                    validation_split=0.2, #20%test
                                                                    seed=1,
                                                                    image_size=size,
                                                                    batch_size=batch_size)

class_names = train_set.class_names #reprend les catégories des images d'entraînements

AUTOTUNE = tf.data.AUTOTUNE #le programme analyse la capacité de l'ordinateur pour être optimal
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

n_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(size[0], size[1], 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(n_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 20
epochs_range = range(epochs)

historique = model.fit(
  train_set,
  validation_data=test_set,
  epochs=epochs)

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
