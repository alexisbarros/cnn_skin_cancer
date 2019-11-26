import os
import cv2 as cv

path_from = './PATH_FROM'
path_to = './PATH_TO/'
count = 0

for img_path in os.listdir(path_from):
    img = cv.imread(os.path.join(path_from, img_path), 0)

    # redimensionar imagem para 256x256
    img = cv.resize(img, (256, 256))

    # equalização clahe
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_equalized = clahe.apply(img)

    # limiarização
    ret, img_thresh = cv.threshold(img_equalized, 127, 255, cv.THRESH_OTSU)

    # destacar lesão
    img_lesion = cv.bitwise_or(img_thresh, img_equalized)

    # salvar lesão segmentada
    cv.imwrite(path_to + img_path, img_lesion)

    # imprimir contador para acompanhar andamento
    print(count)
    count += 1