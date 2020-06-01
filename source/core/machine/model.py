import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.python.keras.callbacks import TensorBoard
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import random
import pickle
import logging
import logging.config
import sys
import time

LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO'
    }
}

logging.config.dictConfig(LOGGING)

DATADIR = "/home/pedro/Área de Trabalho/PetImages"
CATEGORIES = ["Cachorro", "Gato"]
training_data = []
IMG_SIZE = 50


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# def loadData():
#
#     for category in CATEGORIES:  # do dogs and cats
#         path = os.path.join(DATADIR, category)  # create path to dogs and cats
#
#         for img in os.listdir(path):  # iterate over each image per dogs and cats
#
#             img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
#             plt.imshow(img_array, cmap='gray')  # graph it
#             plt.show()  # display!
#
#             break  # we just want one for now so break
#         break  #...and one more!


# def create_training_data():
#     for category in CATEGORIES:  # do dogs and cats
#
#         path = os.path.join(DATADIR, category)  # create path to dogs and cats
#         class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
#
#         for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
#             try:
#                 img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
#                 training_data.append([new_array, class_num])  # add this to our training_data
#             except Exception as e:
#                 pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


# def main():
    # loadData()
    # create_training_data()
    # print(len(training_data))
    # shuffle_train_and_save()
    # train3()
    #open_model_64x3()
#
#
# def shuffle_train_and_save():
#     random.shuffle(training_data)
#     for sample in training_data[:10]:
#         print(sample[1])
#
#     X = []
#     y = []
#     for features, label in training_data:
#         X.append(features)
#         y.append(label)
#
#     X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#
#     pickle_out = open("X.pickle", "wb")
#     pickle.dump(X, pickle_out)
#     pickle_out.close()
#
#     pickle_out = open("y.pickle", "wb")
#     pickle.dump(y, pickle_out)
#     pickle_out.close()


def open_data():
    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle", "rb")
    y = pickle.load(pickle_in)
    X = X/255.0
    return X, y

#
# def model(X, y):
#
#     model = Sequential()
#
#     model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#
#     model.add(Dense(64))
#
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
#     model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
#
#
# def train():
#
#     NAME = "Cats-vs-dogs-64x2-CNN"
#     X,y = open_data()
#     model = Sequential()
#
#     model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#     model.add(Dense(64))
#     model.add(Activation('relu'))
#
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#
#     tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'],
#                   )
#
#     model.fit(X, y,
#               batch_size=32,
#               epochs=10,
#               validation_split=0.3,
#               callbacks=[tensorboard])
#
#
# def train2():
#
#     X,y = open_data()
#
#     dense_layers = [0, 1, 2]
#     layer_sizes = [32, 64, 128]
#     conv_layers = [1, 2, 3]
#
#     for dense_layer in dense_layers:
#         for layer_size in layer_sizes:
#             for conv_layer in conv_layers:
#                 import time
#                 NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#                 print(NAME)
#
#                 model = Sequential()
#
#                 model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 for l in range(conv_layer-1):
#                     model.add(Conv2D(layer_size, (3, 3)))
#                     model.add(Activation('relu'))
#                     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Flatten())
#
#                 for _ in range(dense_layer):
#                     model.add(Dense(layer_size))
#                     model.add(Activation('relu'))
#
#                 model.add(Dense(1))
#                 model.add(Activation('sigmoid'))
#
#                 tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
#                 model.compile(loss='binary_crossentropy',
#                               optimizer='adam',
#                               metrics=['accuracy'],
#                               )
#
#                 model.fit(X, y,
#                           batch_size=32,
#                           epochs=10,
#                           validation_split=0.3,
#                           callbacks=[tensorboard])
#
#
# def train3():
#
#     X, y = open_data()
#
#     dense_layers = [0]
#     layer_sizes = [64]
#     conv_layers = [3]
#
#     for dense_layer in dense_layers:
#         for layer_size in layer_sizes:
#             for conv_layer in conv_layers:
#                 import time
#                 NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#                 print(NAME)
#
#                 model = Sequential()
#
#                 model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 for l in range(conv_layer-1):
#                     model.add(Conv2D(layer_size, (3, 3)))
#                     model.add(Activation('relu'))
#                     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Flatten())
#
#                 for _ in range(dense_layer):
#                     model.add(Dense(layer_size))
#                     model.add(Activation('relu'))
#
#                 model.add(Dense(1))
#                 model.add(Activation('sigmoid'))
#
#                 tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
#                 model.compile(loss='binary_crossentropy',
#                               optimizer='adam',
#                               metrics=['accuracy'],
#                               )
#
#                 model.fit(X, y,
#                           batch_size=32,
#                           epochs=10,
#                           validation_split=0.3,
#                           callbacks=[tensorboard])
#
#     model.save('64x3-CNN.model')
#


def open_model_64x3(path):

    model = tf.keras.models.load_model("source/core/machine/64x3-CNN.model")
    prediction = model.predict([prepare(path)])
    logging.info(time.time())
    logging.info(CATEGORIES[int(prediction[0][0])])

    return CATEGORIES[int(prediction[0][0])]

