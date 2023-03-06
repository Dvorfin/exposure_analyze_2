# Пример выделения маски изображения с возможностью
# настройки диапазонов в простом графическом интерфейсе.

import cv2
import numpy as np
from matplotlib import pyplot as plt
# Функция обработки изображений.
# Она включает в себя преобразование цветовой модели,
# выделение маски по диапазону цветов, совмещение
# оригинального изображения с выделенной маской,
# а также вывод полученных результатов в окне.

def process_img(img, name, color):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_mask = cv2.inRange(hsv_image, color[0], color[1])
    cv2.imshow('mask ' + name, img_mask)

    img_mixed = cv2.bitwise_and(img, img, mask=img_mask)
    cv2.imshow(name, img_mixed)

# Функция, которая будет выполняться при изменении
# значений слайдеров. Сначала заполняем диапазоны,
# а затем запускаем обработку изображений.
#
# Благодаря выделению опердаций над изображениями
# в отдельную функцию, мы можем легко выполнить
# аналогичные действия над ещё одним изображением
# без дублирования кода.

def update(value = 0):
    color_low = (
        cv2.getTrackbarPos('h_min', 'ui'),
        cv2.getTrackbarPos('s_min', 'ui'),
        cv2.getTrackbarPos('v_min', 'ui')
    )

    color_high = (
        cv2.getTrackbarPos('h_max', 'ui'),
        cv2.getTrackbarPos('s_max', 'ui'),
        cv2.getTrackbarPos('v_max', 'ui')
    )

    color = (color_low, color_high)

    process_img(img, 'img', color)
    process_img(rainbow, 'rainbow', color)

# Код, непосредтсвенно выполняемый при запуске скрипта.
# Здесь мы загружаем изображения, которые далее будем
# обрабатывать; а также создаём окно с набором слайдеров.

if __name__ == '__main__':
    # img = cv2.imread('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/Screenshot_2.png')
    # rainbow = cv2.imread('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/rainbow.png')
    #
    #
    # cv2.namedWindow('ui')
    # cv2.createTrackbar('h_min', 'ui',   0, 180, update)
    # cv2.createTrackbar('s_min', 'ui',   0, 255, update)
    # cv2.createTrackbar('v_min', 'ui',   0, 255, update)
    # cv2.createTrackbar('h_max', 'ui', 180, 180, update)
    # cv2.createTrackbar('s_max', 'ui', 255, 255, update)
    # cv2.createTrackbar('v_max', 'ui', 255, 255, update)
    # cv2.imshow('ui', img)
    #
    # update()
    # cv2.waitKey(0)

    #img = cv2.imread('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/Screenshot_3.png')
    img = cv2.imread('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/3-3-29.tif')

    blur = cv2.medianBlur(img, 21)

    # 3 Отображение изображения
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplot(121), plt.imshow(img[:, :, :: - 1]), plt.title('исходное изображение')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur[:, :, :: - 1]), plt.title('Средний результат фильтрации')
    plt.xticks([]), plt.yticks([])
    plt.show()


