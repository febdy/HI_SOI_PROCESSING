# python train_network.py --dataset dataset --model face_ex.model

import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

epochs = 25
init_lr = 1e-3
bs = 32  # batch_size

data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == 'face' else 0
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

train_y = to_categorical(train_y, num_classes=2)
test_y = to_categorical(test_y, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=init_lr, decay=init_lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(aug.flow(train_x, train_y, batch_size=bs), validation_data=(test_x, test_y),
                        steps_per_epoch=len(train_x) // bs, epochs=epochs, verbose=1)

model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
n = epochs
plt.plot(np.arange(0, n), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, n), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
