import os
from PIL import Image

path_from = './PATH_FROM'
path_to = './PATH_TO/'
count = 0

for img_path in os.listdir(path_from):
    img = Image.open(os.path.join(path_from, img_path))

    # pegar tamanho da image
    w, h = img.size

    # calculando o tamanho do quadrado
    area = (0, 0, w, h)
    if h < w:
       area = (((w - h) / 2), 0, (w - ((w - h) / 2)), h)
    elif w < h:
        area = (0, ((h - w) / 2), w, (w + ((h - w) / 2)))

    # cortar imagem no centro
    cropped_img = img.crop(area)

    # salvar img
    cropped_img.save(path_to + img_path)

    # imprimir contador para acompanhar andamento
    print(count)
    count += 1
