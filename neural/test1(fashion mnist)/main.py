import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import numpy as np

#загрузка данных
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

#преобразов размерности изображений
x_train=x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
#ормализация данных
x_train =x_train/255
x_test = x_test / 255

y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test, 10)
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']


#Создание последовательной модели
model = Sequential()
model.add(Dense(800,input_dim=784,activation="relu"))
model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())

# Обучение модели(размер мини-выборки = 200, эпох 100) и сохранение модели
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)
model.save('fashion_mnist.h5')

prediction = model.predict(x_train)

print(prediction[0])

print(np.argmax(prediction[0]))

print(np.argmax(y_train[0]))

scores = model.evaluate(x_test, y_test, verbose=1)

print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1]*100,4))
