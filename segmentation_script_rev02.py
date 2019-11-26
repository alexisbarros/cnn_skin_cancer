import os
import cv2 as cv

path_from = './dataset_crop/melanoma'
path_to = './dataset_lesion_mask/'
count = 0

trainOrValidation = 'train/'

for img_path in os.listdir(path_from):
    img = cv.imread(os.path.join(path_from, img_path), 0)

    # redimensionar imagem para 256x256
    img = cv.resize(img, (256, 256))

    # equalização clahe
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_equalized = clahe.apply(img)

    # limiarização
    ret, img_thresh = cv.threshold(img_equalized, 127, 255, cv.THRESH_OTSU)

    if count >= 360:
        trainOrValidation = 'validation/'

    # salvar lesão segmentada
    cv.imwrite(path_to + trainOrValidation + 'melanoma/' + img_path, img_thresh)

    # imprimir contador para acompanhar andamento
    print(count)
    count += 1