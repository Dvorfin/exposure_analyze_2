
# модуль для преобразования изобрадения в чб, обработки
# фильтром гаусса и сохранения результатов
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cv2 as cv
#mpl.use('nbagg')


pictures_path = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/' # path to photos

photo_names = [str(i) + '.tif' for i in range(6)]   #  names of photos

path = pictures_path + '0.tif'

# path = 'C:/Users/Root/Desktop/Exposure_experiment/second/gs/GS-2/2-1.tif'


def split_for_3_channels(image):
    b, g, r = cv.split(image)
    cv.imshow('blue', b)
    cv.imshow('green', g)
    cv.imshow('red', r)
    cv.waitKey(0)
    cv.destroyAllWindows()


def load_picture(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    cv.imshow('picture', img)

    img_blur_3 = cv.GaussianBlur(img, (25, 25), 9) # наиболее приближено к первы тестам
    cv.imshow('img_blur_3', img_blur_3)

    cv.imwrite('test.tif', img_blur_3)


    cv.waitKey(0)
    cv.destroyAllWindows()

    #   cv2.imwrite('graygirl.jpg', img) - for saving picture

counter = 0
x1 = 0
x2 = 0
y1 = 0
y2 = 0
img = 0
def show_picture(path):
    global img
    img = cv.imread(path, cv.IMREAD_COLOR)
    #img = plt.imread(path) # считывание через matplotlib

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('picture')
    fig.set_figwidth(12)  # ширина и
    fig.set_figheight(8)  # высота "Figure"

    mng = plt.get_current_fig_manager() # блокирует изменение размера окна
    mng.window.resizable(False, False)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


def increment():
    global counter
    counter += 1
    print(counter)
    if counter >= 2:
        plt.close('all')
        counter = 0

        global img
        img = img[x1:x2, y1:y2]
        cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/' + 'test.tif', img)

def crop_points(x, y):
    global x1, x2, y1, y2
    if counter == 0:
        x2, y2 = x, y
    if counter == 1:
        x1, y1 = x, y
    print(x1, y1, x2, y2)

def onclick(event):
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
    #     event.button, event.x, event.y, event.xdata, event.ydata))
    global img
    print(f'x: {round(event.xdata)}    | y: {round(event.ydata)}')
    increment()
    crop_points(round(event.xdata), round(event.ydata))


if __name__ == '__main__':

    #print(pictures_path + photo_names[0])

    # load_picture(path)
    show_picture(path)

