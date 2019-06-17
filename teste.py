from PIL import Image, ImageEnhance
import os

# Directories array to loop for
directories = ['benign/', 'malignant/']

# Dataset and label
dataset = []

# Loop in directories
for dir in directories:

    # Loop in images
    for index, filename in enumerate(os.listdir('dataset/' + dir)):

        print(index)

        # Get image
        img = Image.open('dataset/' + dir + filename)

        # Brightness
        img = ImageEnhance.Brightness(img).enhance(1.5)

        img.save('teste.png')

        break
    break
