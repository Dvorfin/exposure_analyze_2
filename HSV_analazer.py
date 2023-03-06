
# модуль для обрезки изображения по периметру и сохранению обрезанных изображений

import sys
import numpy as np
import cv2 as cv
import math
#
# hsv_min = np.array((98, 163, 0), np.uint8)
# hsv_max = np.array((255, 176, 254), np.uint8)

# hsv_min = np.array((65, 166, 171), np.uint8)
# hsv_max = np.array((139, 172, 242), np.uint8)
#
# hsv_min = np.array((87, 109, 0), np.uint8)
# hsv_max = np.array((97, 181, 255), np.uint8)
#
# hsv_min = np.array((66, 158, 223), np.uint8)
# hsv_max = np.array((97, 170, 225), np.uint8)
# #
# hsv_min = np.array((51, 51, 0), np.uint8)
# hsv_max = np.array((255, 255, 225), np.uint8)

# hsv_min = np.array((98, 160, 0), np.uint8)
# hsv_max = np.array((255, 171, 254), np.uint8)

# hsv_min = np.array((98, 165, 0), np.uint8)
# hsv_max = np.array((107, 172, 255), np.uint8)

# hsv_min = np.array((0, 0, 0), np.uint8)
# hsv_max = np.array((0, 255, 255), np.uint8)

hsv_min = np.array((85, 0, 0), np.uint8)
hsv_max = np.array((107, 172, 255), np.uint8)


# на вход подать изображение и область (распознанную) по которой обрезать
def crop_rect(img, rect):
    # rect ->  ((центр прямогуольника), (размер прямоугольника), угол наклона)
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)    # формирует матрицу поворота
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))        # вращает изображение

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop


if __name__ == '__main_8_':
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/scan0002.tif'  # имя файла, который будем анализировать
    #fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/crop_8.tif'  # имя файла, который будем анализировать
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/0.tif'
    img = cv.imread(fn)

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    y_range, x_range, _ = img.shape  # задаем рамзеры картинки
    cv.resizeWindow('result', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cropped_img_num = 1  # номер изображения кропнутого

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат

        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        # if area > 6_00_000:
        #     print(area)
        if 6_000_000 < area < 8_000_000:  # если площадь прямогульника больше  < 9_000_000
            print(f'rect params: {rect}')
            print(area)  # примерно должно быть 6_500_500
            cropped_img = crop_rect(img, rect)

            cv.drawContours(img, [box], 0, (255, 0, 0), 8)      # отрисовка прямогуольников размером больше 700_000
            #cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/crop_new' + f'{cropped_img_num}.tif', cropped_img)

            cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/crop_new' + f'{cropped_img_num}.tif', cropped_img,
                       ((int(cv.IMWRITE_TIFF_RESUNIT), 2,
                         int(cv.IMWRITE_TIFF_COMPRESSION), 1,
                         int(cv.IMWRITE_TIFF_XDPI), 600,
                         int(cv.IMWRITE_TIFF_YDPI), 600)))


            cropped_img_num += 1

    cv.imshow('result', img)

    cv.waitKey()
    cv.destroyAllWindows()





# import cv2
# import numpy as np
#
#
# if __name__ == '__main__':
#    def callback(*arg):
#        print (arg)
#
# cv2.namedWindow( "result" )
#
# fn = 'C:/Users/Root/Desktop/Exposure_experiment/Screenshot_3.png'  # имя файла, который будем анализировать
# img = cv2.imread(fn)
# hsv_min = np.array((0, 0, 0), np.uint8)
# hsv_max = np.array((100, 255, 255), np.uint8)
#
# while True:
#     #flag, img = cap.read()
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
#     thresh = cv2.inRange(hsv, hsv_min, hsv_max)
#
#     cv2.imshow('result', thresh)
#
#     ch = cv2.waitKey(5)
#     if ch == 27:
#         break
#
#
# cv2.destroyAllWindows()












# -------------------------------------------
# поиск по HSV


import cv2
import numpy as np
#import video

if __name__ == '__main__':
    def nothing(*arg):
        pass


def HSV_analyzer():

    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/first exp cropped by python/2-1.tif'  # имя файла, который будем анализировать
    #fn = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/p1.jpg'
    fn = 'C:/Users/Root/Desktop/Exposure_experiment/2-1.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/cropped/3-4.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/Screenshot_8.png'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/scan0002.tif'
    #fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/3-3-29.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/overlaped.tif'

    img = cv2.imread(fn)
    img = cv2.medianBlur(img, 21)

    #cv2.imshow(img)
    print(img.shape)
    y_range, x_range, _ = img.shape  # задаем рамзеры картинки

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)  # создаем главное окно
    cv2.resizeWindow('result', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза
    cv2.namedWindow("settings")  # создаем окно настроек

    cap = img

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
    crange = [0, 0, 0, 0, 0, 0]
    cv2.resizeWindow('settings', 700, 140)

    while True:
        #######flag, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv2.getTrackbarPos('h1', 'settings')
        s1 = cv2.getTrackbarPos('s1', 'settings')
        v1 = cv2.getTrackbarPos('v1', 'settings')
        h2 = cv2.getTrackbarPos('h2', 'settings')
        s2 = cv2.getTrackbarPos('s2', 'settings')
        v2 = cv2.getTrackbarPos('v2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv2.inRange(hsv, h_min, h_max)


        #blur = cv2.medianBlur(thresh, 21)
       # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
        cv2.imshow('result', thresh)

        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
# -------------------------------------------

def binarization():
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/overlaped.tif'
    #fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/3-3-29.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/crop_new1.tif'
    dn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/cropped/3-3.tif'
    img = cv2.imread(fn)

    #img = cv.GaussianBlur(img, (25, 25), 9)
    #img = cv2.medianBlur(img, 21)

    y_range, x_range, _ = img.shape

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)  # создаем главное окно
    cv2.resizeWindow('result', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза
    cv2.namedWindow("settings")  # создаем окно настроек


    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('low', 'settings', 0, 255, nothing)
    cv2.createTrackbar('high', 'settings', 0, 255, nothing)

    cv2.resizeWindow('settings', 700, 140)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    while True:
        #######flag, img = cap.read()


        # считываем значения бегунков
        h1 = cv.getTrackbarPos('low', 'settings')
        s1 = cv.getTrackbarPos('high', 'settings')

        # накладываем фильтр на кадр в модели HSV

        #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 1001, 0 + h1 // 10)
        #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 501, 0 + h1 // 10)

        # ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_BINARY)
        #
        # kernel = np.ones((25, 25), dtype=np.uint8)
        # thresh = cv.erode(thresh, kernel)
        # thresh = cv2.dilate(thresh, kernel)

        ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_BINARY)
        #ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_BINARY_INV)
       # ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_TRUNC)
      #  ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_TOZERO)
       # ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_TOZERO_INV)

       # thresh = cv2.medianBlur(thresh, 21)
        # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
        cv2.imshow('result', thresh)

        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()

#HSV_analyzer()

binarization()