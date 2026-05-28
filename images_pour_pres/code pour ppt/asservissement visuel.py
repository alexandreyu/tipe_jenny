import time
import cv2
import numpy as np
import serial
import tensorflow as tf

# --- Configuration de la liaison USB (Série) ---
# Ton port COM4 à 115200 bauds
arduino = serial.Serial(port='COM4', baudrate=115200, timeout=1)
time.sleep(5)  # Pause de sécurité pour laisser l'Arduino redémarrer

# --- Chargement de TON réseau de neurones ---
print("Chargement du modèle...")
model = tf.keras.models.load_model('reseau.h5')
classes = ['Droite', 'Gauche', 'Milieu']

# --- Configuration du flux vidéo ---
# Utilisation de ton URL de flux
url_flux = "http://192.168.144.50:81/stream"

print(f"Connexion au flux vidéo : {url_flux} ...")
cap = cv2.VideoCapture(url_flux)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux vidéo. Vérifie l'adresse IP.")
    arduino.write(b'S')
    exit()

print("Flux connecté ! Initialisation terminée.")

try:
    while True:
        # 1. Capture de l'image en direct depuis le flux
        ret, frame = cap.read()

        if not ret:
            print("Erreur : Impossible de lire l'image du flux.")
            arduino.write(b'S')
            time.sleep(0.5)
            continue

        try:
            # 2. Prétraitement de l'image
            # OpenCV utilise le format BGR, ton modèle a besoin de RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Redimensionnement aux dimensions exactes de ton modèle (160 de large, 120 de haut)
            # /!\ Attention : cv2.resize prend (Largeur, Hauteur) donc (160, 120)
            img_resized = cv2.resize(img_rgb, (160, 120))

            # Conversion en float et normalisation comme dans ton code précédent
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension "batch"

            # 3. Prédiction par le réseau de neurones
            predictions_brutes = model.predict(img_array, verbose=0)

            # Application de Softmax car ton modèle utilise 'from_logits=True' à l'entraînement
            probabilites = tf.nn.softmax(predictions_brutes).numpy()

            classe_index = np.argmax(probabilites[0])
            classe_index = np.argmax(probabilites[0])
            classe_predite = classes[classe_index]
            confiance = probabilites[0][classe_index]

            print(f"Prediction : {classe_predite} ({confiance * 100:.2f}%)")

            # Seuil de confiance à 70%
            if confiance > 0.70:
                # 4. Envoi de l'ordre à l'Arduino
                if classe_predite == 'Gauche':
                    arduino.write(b'R')  # reculer
                elif classe_predite == 'Droite':
                    arduino.write(b'A')  # avancer
                else:
                    arduino.write(b'S')  # Stop
            else:
                print("Confiance trop faible -> Arrêt")
                arduino.write(b'S')

            # 5. BONUS VISUEL : Affiche la vidéo sur ton ordi pour voir ce que l'IA fait
            texte_ecran = f"{classe_predite} ({confiance * 100:.0f}%)"
            cv2.putText(frame, texte_ecran, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Vision du Robot", frame)

        except Exception as e:
            print(f"Erreur lors du traitement/IA : {e}")
            arduino.write(b'S')

        # Une micro-pause pour éviter de surcharger le processeur
        # (L'affichage cv2.waitKey obligatoire pour rafraîchir la fenêtre)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Arrêt demandé par l'utilisateur.")

finally:
    # --- Nettoyage à la fermeture ---
    print("Fermeture et sécurité...")
    arduino.write(b'S')  # Arrêt des moteurs par sécurité
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
    print("Programme arrêté proprement.")