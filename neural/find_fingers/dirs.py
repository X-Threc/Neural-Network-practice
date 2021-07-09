import os, glob
from PIL import Image

# выполняет функцию изменения размера изображений из директории
def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)


if __name__ == '__main__':
    train_img_list = glob.glob("fingers/train2/*.jpg")
    i=1
    for img in train_img_list:
        resize_image(input_image_path=img,
                     output_image_path=img,
                     size=(128, 128))
        print(i)
        i += 1