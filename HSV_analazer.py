
# модуль для обрезки изображения по периметру и сохранению обрезанных изображений

import sys
# import numpy as np
# import cv2 as cv
#
# hsv_min = np.array((0, 0, 0), np.uint8)
# hsv_max = np.array((100, 255, 255), np.uint8)
#
# if __name__ == '__main__':
#     fn = 'C:/Users/Root/Desktop/Exposure_experiment/Screenshot_6 (1).png'  # имя файла, который будем анализировать
#     img = cv.imread(fn)
#
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
#     thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
#     contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     # перебираем все найденные контуры в цикле
#     for cnt in contours0:
#         rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
#         box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
#         box = np.int0(box)  # округление координат
#         cv.drawContours(img, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
#
#     cv.imshow('contours', img)  # вывод обработанного кадра в окно
#
#     cv.imwrite(fn, img)
#     cv.waitKey()
#     cv.destroyAllWindows()





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



fn = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/first exp cropped by python/2-1.tif'  # имя файла, который будем анализировать
#fn = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/p1.jpg'
fn = 'C:/Users/Root/Desktop/Exposure_experiment/2-1.tif'
fn = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/cropped/3-4.tif'
fn = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/cropped/3-4.tif'
fn = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/3-16-g.tif'

img = cv2.imread(fn)

y_range, x_range, _ = img.shape  # задаем рамзеры картинки

cv2.namedWindow("result", cv2.WINDOW_NORMAL)  # создаем главное окно
cv2.resizeWindow('result', int(x_range // 5), int(y_range // 5))  # уменьшаем картинку в 3 раза
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
   # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
    cv2.imshow('result', thresh)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()
# -------------------------------------------