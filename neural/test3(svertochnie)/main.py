import numpy as np
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from PIL import Image


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # стандартизация входных данных
# x_train = x_train / 255
# x_test = x_test / 255
#
# y_train_cat = keras.utils.to_categorical(y_train, 10)
# y_test_cat = keras.utils.to_categorical(y_test, 10)
#
# x_train = np.expand_dims(x_train, axis=3)
# x_test = np.expand_dims(x_test, axis=3)
# print( x_train.shape )
#
# model = keras.Sequential([
#     Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D((2, 2), strides=2),
#     Conv2D(64, (3,3), padding='same', activation='relu'),
#     MaxPooling2D((2, 2), strides=2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10,  activation='softmax')
# ])
#
# print(model.summary())
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

img_path = '1.jpg'
img = image.load_img(img_path)
img = image.load_img(img_path, target_size=(224, 224))

# Преобразуем картинку в массив
x = np.array(img)


# загружаем модель vgg16
model = keras.applications.VGG16()
x = keras.applications.vgg16.preprocess_input(x)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)
# прогоняем через сеть
res = model.predict(x)
print(decode_predictions(res, top=3))
print(np.argmax(res))