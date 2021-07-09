from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

model = load_model('fashion_mnist.h5')

model.summary()
img_path = 't.jpg'
img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
# Преобразуем картинку в массив
x = np.asarray(img)


# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x =x / 255


prediction = model.predict(x)
print(prediction)
prediction = np.argmax(prediction)
print("Номер класса:", prediction)
print("Название класса:", classes[prediction])