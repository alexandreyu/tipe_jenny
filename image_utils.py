from PIL import Image
import tensorflow as tf

# Résolution : 160 x 120


def grayscale(path):
    img = Image.open(path)
    out = img.convert("L")
    out.save("gs_" + path)


def load_png(path, channels):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=channels)
    return img


# grayscale("cat_test.png")

print(load_png("gs_cat_test.png", 1))
