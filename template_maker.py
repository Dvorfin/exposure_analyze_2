
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

path = 'C:/Users/Root/Desktop/template_maker/'
names = ['Screenshot_1.png', 'Screenshot_2.png', 'Screenshot_3.png',
         'Screenshot_4.png', 'Screenshot_5.png', 'Screenshot_6.png',
         'Screenshot_7.png']

# function calculates minimal shape of all images and
# overlap all images to minimal shape
def overlap_images(imgs_path, imgs_name):

    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   #  converitng to RGB
    # if showing into matplotlib then need to convert into RGB
    # if showing into opencv then no need in converting

    imgs = [cv.imread(imgs_path + name, cv.IMREAD_COLOR) for name in imgs_name]     # list of images
    img_shapes = [img.shape for img in imgs]        # list of tuples including images shapes (h, w, pixel)
    min_height = min(img_shapes, key=lambda item: item[0])[0]  # minimal height of images
    min_width = min(img_shapes, key=lambda item: item[1])[1]    # minumal width of images

    # imgs = [cv.GaussianBlur(img, (25, 25), 5) for img in imgs]    # gauss filter for each image

    # reshaping all images to size of minimal image
    imgs = [img[0:min_height, 0:min_width] for img in imgs]

    print(f'Minimal image shape:\ny: {min_height}\tx: {min_width}')

    return sum(imgs)        # returning sum of all images


if __name__ == '__main__':
    path = 'C:/Users/Root/Desktop/template_maker/'
    path = 'C:/Users/Root/Desktop/Exposure_experiment/2/'
    names = [f'2-{i}.tif' for i in range(1, 17)]


    cv.namedWindow("result", cv.WINDOW_NORMAL)      # resize window
    cv.resizeWindow('result', int(3766 / 9), int(6393 / 9))

    res = overlap_images(path, names)

    # res = cv.cvtColor(res, cv.COLOR_BGR2GRAY) # converting to gs

    #blur = cv.GaussianBlur(res, (25, 25), 5)  # применение фильтра гаусса
    cv.imshow('result', res)

    cv.waitKey(0)
    print(res.shape)


    # img = cv.imread(path + names[5], cv.IMREAD_COLOR)     # reading in default
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   #  converitng to RGB


