from shader import *


def category_test(image, n):
    center = process_image(image)

    for i in range(1, n + 1):
        if center[1] < image.shape[1] * i/n:
            return i


def label_folder(path, outpath):
    category_amount = len(os.listdir(outpath))
    for i in os.listdir(path):
        image = cv.imread(path + "/" + i)
        category = category_test(image, category_amount)
        cv.imwrite(outpath + "/" + str(category) + "/" + i, image)


label_folder("to_label", "to_dataset")

