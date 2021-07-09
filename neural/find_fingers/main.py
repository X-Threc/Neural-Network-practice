import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from skimage import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model
from PIL import Image

print("Загрузка...")
print(device_lib.list_local_devices())


pd=tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(pd[0],True)

train_img_list = glob.glob("fingers/train2/*.jpg")
test_img_list = glob.glob("fingers/test2/*.jpg")
print(len(train_img_list), len(test_img_list), sep = '\n')
print("Загрузка изображений...")

# функция для получения фотографий для тренировки и теста
# def import_data():
#     train_img_data = []
#     test_img_data = []
#     train_label_data = []
#     test_label_data = []
#     i = 1
#     for img in train_img_list:
#         img_read = io.imread(img)
#         train_img_data.append(img_read)
#         train_label_data.append(img[15])
#         print(i)
#         i+=1
#
#     for img in test_img_list:
#         img_read = io.imread(img)
#         test_img_data.append(img_read)
#         test_label_data.append(img[14])
#         print(i)
#         i+=1
#
#     return np.array(train_img_data), np.array(test_img_data), np.array(train_label_data), np.array(test_label_data)
#
#
# xtrain, xtest, ytrain, ytest = import_data()
# print(xtrain.shape)
# xtrain = xtrain.reshape(xtrain.shape[0], 128, 128, 3)
# xtest = xtest.reshape(xtest.shape[0], 128, 128, 3)
# xtrain = xtrain / 255
# xtest = xtest / 255
#
# ytrain = tf.keras.utils.to_categorical(ytrain, num_classes = 6)
# ytest = tf.keras.utils.to_categorical(ytest, num_classes = 6)
#
#
# print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
#
# # разбиение тренировочной выборки на тренировочную выборку и выборку для проверки
# x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size = 0.20, random_state = 7, shuffle = True)
# x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(x_train, y_train, test_size = 0.20, random_state = 7, shuffle = True)
#
#
# # создание свёрточной модели сети
# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape = (128,128, 3), activation = 'relu'))
# model.add(Conv2D(32, (3,3), activation = 'relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(64, (3,3), activation = 'relu'))
# model.add(Conv2D(64, (3,3), activation = 'relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128, (3,3), activation = 'relu'))
# model.add(Conv2D(128, (3,3), activation = 'relu'))
#
# model.add(Flatten())
# model.add(Dropout(0.60))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dropout(0.60))
# model.add(Dense(6, activation = 'softmax'))
#
# model.summary()
# model.compile('SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# # обучение нейронной сети
# model.fit(x = x_train, y = y_train, batch_size = 5, epochs = 6, validation_data = (x_train_val, y_train_val))
# model.save('model_finger1.h5')
#
#
# pred = model.evaluate(xtest,
#                       ytest,
#                       batch_size = 128)
# print("Доля верных ответов на тестовых данных, в процентах: ", pred[1]*100)
# pred = model.evaluate(x_test_val,
#                       y_test_val,
#                       batch_size = 128)
# print("Доля верных ответов на тестовых данных, в процентах: ", pred[1]*100)


model = load_model('model_finger1.h5')
img_path = '5 (33).jpg'
original_image = Image.open(img_path)
img = original_image.resize((128,128))

# Преобразуем картинку в массив
x = np.array(img)
print(x.shape)
x=np.reshape(x,(128,128,3))
x = np.expand_dims(x, axis=0)

print(x.shape)
x=x/255
prediction = model.predict(x)
prediction = np.argmax(prediction)
print("Номер класса:", prediction)

