import cv2 as cv
import numpy as np
import tensorflow as tf
import serial  # Importation pour la liaison série
import time

# 1. Configuration de la liaison série avec l'Arduino Nano
# Remplace 'COM3' par le vrai port de ton Arduino (ex: 'COM4', '/dev/ttyUSB0' sur Mac)
try:
    arduino = serial.Serial(port='COM3', baudrate=9600, timeout=0.1)
    time.sleep(2)  # Pause de 2 secondes pour laisser l'Arduino redémarrer proprement
    print("Connexion série avec l'Arduino Nano réussie !")
except Exception as e:
    print(f"Erreur de connexion Arduino : {e}")
    arduino = None

model = tf.keras.models.load_model('ton_modele_sauvegarde.h5')
cap = cv.VideoCapture("http://192.168.1.50/stream")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- (Ton pré-traitement d'image ici) ---
    img_resized = cv.resize(frame, (160, 120)) # Format QQVGA
    img_array = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # --- PRÉDICTION ---
    prediction = model.predict(img_batch, verbose=0)
    classe_predite = int(np.argmax(prediction)) # 0, 1, ou 2 selon tes catégories

    # --- ENVOI DE LA COMMANDE À L'ARDUINO ---
    if arduino is not None:
        # On envoie le numéro de la classe suivi d'un saut de ligne '\n'
        commande = f"{classe_predite}\n"
        arduino.write(commande.encode('utf-8')) # .encode() transforme le texte en octets

    # --- AFFICHAGE ---
    cv.imshow("Suivi", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

if arduino: arduino.close()
cap.release()
cv.destroyAllWindows()