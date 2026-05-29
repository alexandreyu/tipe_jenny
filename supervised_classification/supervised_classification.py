import cv2; #image reader
import os


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


path_right = "../dataset/droite/"
path_middle = "../dataset/milieu/"
path_left = "../dataset/gauche/"

images_right = []
images_left = []
images_middle = []


def append_in_images_array(tab, path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filename))
            tab.append(img)

#Matrix -> Vector
def pre_processing_images(image_list):
    processed = []
    for img in image_list:
        img_resized = cv2.resize(img, (160,120))
        img_flat = img_resized.flatten()
        processed.append(img_flat)
    return processed



append_in_images_array(images_right, path_right)
append_in_images_array(images_left, path_left)
append_in_images_array(images_middle, path_middle)

right_processed_images = pre_processing_images(images_right)
left_processed_images = pre_processing_images(images_left)
middle_processed_images = pre_processing_images(images_middle)



X = np.array(right_processed_images + left_processed_images + middle_processed_images) #Data
Y = ['droite']*len(images_right) + ['gauche']*len(images_left) + ['milieu']*len(images_middle) #Real labels

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, criterion='gini', max_depth=10000)

# Calculate the model
model.fit(X_train, Y_train)

# Test the model
predictions = model.predict(X_test)
print(f"Précision du modèle : {accuracy_score(Y_test, predictions) * 100} %")



#Tests images figure
plt.figure(figsize=(15,10))
for i in range(9):
    plt.subplot(3,3,i+1)

    img = X_test[i].reshape((120,160,3))

    plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.title(f"Pred : {predictions[i]}\nVrai : {Y_test[i]}")

    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

