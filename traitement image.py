import time
import cv2
import numpy as np
import serial
import tensorflow as tf

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=1)
time.sleep(5)

url_flux = "http://10.211.228.50:81/stream"
model = tf.keras.models.load_model('reseau.h5')
classes = ['milieu', 'droite', 'milieu']

cap = cv2.VideoCapture(url_flux)
while True:
    ret, frame = cap.read()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (120, 160))
    img_array = img_resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    predictions_brutes = model(img_array, training=False).numpy()

    probabilites = tf.nn.softmax(predictions_brutes).numpy()

    classe_index = np.argmax(probabilites[0])
    classe_predite = classes[classe_index]
    confiance = probabilites[0][classe_index]

    print(f"Prediction : {classe_predite} {confiance * 100}%")
    if confiance > 0.70:
        if classe_predite == 'droite':
            arduino.write(b'R')  # reculer
        elif classe_predite == 'gauche':
            arduino.write(b'A')  # avancer
        else:
            arduino.write(b'S')  # Stop
    else:
        print("Confiance trop faible -> Arrêt")
        arduino.write(b'S')

    texte_ecran = f"{classe_predite} ({confiance * 100:.0f}%)"
    cv2.putText(frame, texte_ecran, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Vision du Robot", frame)



    # (L'affichage cv2.waitKey obligatoire pour rafraîchir la fenêtre)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
