
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

#path = 'C:/Users/Root/Desktop/template_maker/'
#names = ['Screenshot_1.png', 'Screenshot_2.png', 'Screenshot_3.png',
#         'Screenshot_4.png', 'Screenshot_5.png', 'Screenshot_6.png',
#         'Screenshot_7.png']


# function calculates minimal shape of all images and
# overlap all images to minimal shape
def overlap_images(imgs_path, imgs_name):

    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   #  converitng to RGB
    # if showing into matplotlib then need to convert into RGB
    # if showing into opencv then no need in converting

    imgs = [cv.imread(imgs_path + name, cv.IMREAD_COLOR) for name in imgs_name]     # list of images
    img_shapes = [img.shape for img in imgs]        # list of tuples including images shapes (h, w, pixel)

    print(img_shapes)

    min_height = min(img_shapes, key=lambda item: item[0])[0]  # minimal height of images
    min_width = min(img_shapes, key=lambda item: item[1])[1]    # minumal width of images

    imgs = [cv.GaussianBlur(img, (9, 9), 0) for img in imgs]    # gauss filter for each image

    # reshaping all images to size of minimal image
    imgs = [img[0:min_height, 0:min_width] for img in imgs]
    imgs = [np.uint16(img) for img in imgs]

    print(f'Minimal image shape:\ny: {min_height}\tx: {min_width}')

    res = np.uint8(sum(imgs) // len(imgs))
    return res       # returning sum of all images


def make_overlaped_template():
    path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/rot/'
    names = [f'{i}.tif' for i in range(0, 10)]

    res = overlap_images(path, names)
    y, x, _ = res.shape
    cv.namedWindow("result", cv.WINDOW_NORMAL)  # resize window
    cv.resizeWindow('result', x // 3, y // 3)

    plt.imshow(cv.imread(path + names[0], 0))
    plt.show()
    # res = cv.cvtColor(res, cv.COLOR_BGR2GRAY) # converting to gs

    blur = cv.GaussianBlur(res, (25, 25), 5)  # применение фильтра гаусса
    cv.imshow('result', blur)

    cv.waitKey(0)
    print(res.shape)

    print(f'Overlaped img saved: {path}')
    cv.imwrite(path + 'overlaped_1.tif', blur)

    # img = cv.imread(path + names[5], cv.IMREAD_COLOR)     # reading in default
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   #  converitng to RGB


def calc_avg_square():
    square = cv.imread('C:/Users/Root/Desktop/template_maker/3-squares.tif', 0)
    print(square.shape)

    sq_1 = square[0:][0:684]
    sq_2 = square[0:][684:1368]
    sq_3 = square[0:][1368:]

    imgs = [cv.GaussianBlur(img, (25, 25), 5) for img in [sq_1, sq_2, sq_3]]

    res = sum(imgs)

    cv.imshow('1', sq_1)
    cv.imshow('2', sq_2)
    cv.imshow('3', sq_3)
    cv.imshow('overlaped', res)
    cv.imwrite('C:/Users/Root/Desktop/template_maker/res.tif', res)
    cv.waitKey(0)


def old_main():
    template = np.zeros([684, 684], dtype=np.uint8)
    template.fill(121)

    img = cv.imread('C:/Users/Root/Desktop/template_maker/res.tif', 0)

    # plt.title(f'bungard')
    # y = [template[342][i] - img[342][i] for i in range(684)]
    # x = [i for i in range(684)]
    # plt.plot(x, y)
    # plt.show()
    y = 342
    x = 342
    # наверное можно исползовать какую нибудь геометрическую или арифметическую прогрессию

    for i in range(y):
        lim = 2
        deli = 2
        pixel = 255
        for j in range(x):
            if j > lim:
                pixel -= 1
                lim += deli

            if pixel <= 0:
                pixel = 255

            template[i][j] += pixel

    for i in range(y):
        lim = 4
        deli = 9
        pixel = 255
        for j in range(x):
            if j > lim:
                pixel -= 1
                lim += deli

            if pixel <= 0:
                pixel = 255

            template[j][i] += pixel

    bitwiseAnd = cv.bitwise_and(template, img)
    cv.imshow("AND", bitwiseAnd)

    print(template.shape)
    cv.imshow('template', template)


if __name__ == '__main__':
    path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/rot/'
    # names = [f'2-{i}.tif' for i in range(1, 17)]
    names = [f'{i}.tif' for i in range(0, 10)]

    make_overlaped_template()





