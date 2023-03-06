# построение графиков вдоль широкой грани изображения
# и попытка интерполировать кривую

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/scnitto/edge_slice/3-4.tiff'
path =  'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/3-3-g-ps.tif'
# path = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/edge_slice_first/2-1.tif'
path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/photoshop_contrast/3-3-g.tif'



def calc_mistake():
    pass


# y_range, x_range = img.shape  # задаем рамзеры картинки
# print(f'image size:  {x_range} x {y_range}')


def nothing(*arg):
    pass


# cv.namedWindow("settings")  # создаем окно настроек
# cv.resizeWindow('settings', 400, 80)
# cv.createTrackbar('x', 'settings', 0, x_range, nothing)
# cv.createTrackbar('y', 'settings', 0, y_range, nothing)
#
# cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно и ресайзим его
# cv.resizeWindow('result', int(x_range // 5), int(y_range // 5))  # уменьшаем картинку в 3 раза


prev_s1 = 0  # предыдущее состояние горизонтальной линни


#  функция отрисовки плота
def show_plot(image, x_r, y_r, line):
    x = [i for i in range(x_r)]
    # rotated_image = np.rot90(image)
    rotated_image = image

    y = [rotated_image[line][i] for i in range(x_r)]

    plt.plot(x, y, label=str(line))
    plt.legend()
    plt.show()

# функция отрисовки графика шаблона и второй картинки
def show_two_plots(template, image, x_r, y_r, line):
    y_t, x_t = template.shape
    max_range = min(x_t, x_r)
    x = [i for i in range(max_range)]
    y = [image[line][i] for i in range(max_range)]


    plt.subplot(3, 1, 1)    # plot of bungard
    plt.title(f'Template bungard')
    plt.ylabel('value')
    plt.xlabel('tone')

    line_t = y_t // 2
    y_t = [template[line_t][i] for i in range(max_range)]
    x_t = [i for i in range(max_range)]
    plt.plot(x_t, y_t, label=f'line: {line_t}')
    plt.legend()



    plt.subplot(3, 1, 2)    # plot of image
    plt.title(f'exp')
    plt.ylabel('value')
    plt.xlabel('tone')
    plt.plot(x, y, label=f'line: {str(line)}')
    plt.legend()



    plt.subplot(3, 1, 3)    # plot of difference between bungard and image
    y_d = [abs(y[i] - y_t[i]) for i in range(max_range)]
    #y_d = [abs(y_t[i] - y[i]) for i in range(max_range)]
    x_d = [i for i in range(len(y_d))]
    plt.title('difference')
    plt.plot(x_d, y_d)
    plt.show()


def calc_diff(template, image, exp_name=None, destination=None):
    y, x = image.shape
    y_t, x_t = template.shape

    min_range = min(x, x_t)  # минимальная ширина картинки

    plt.ion()
    plt.figure(figsize=(20, 12))

    x_template = [i for i in range(min_range)]
    line_t = y_t // 2
    y_template = [template[line_t][i] for i in range(min_range)]
    plt.subplot(3, 1, 1)
    plt.title(f'bungard')
    plt.plot(x_template, y_template)

    x_img = [i for i in range(min_range)]  # x стоял
    line_img = y // 2
    y_img = [image[line_img][i] for i in range(min_range)]
    plt.subplot(3, 1, 2)
    if exp_name:
        plt.title(f'{exp_name}')
    else:
        plt.title(f'Exp')
    plt.plot(x_img, y_img)

    plt.subplot(3, 1, 3)  # plot of difference between bingard and image
    y_d = [abs(int(y_template[i]) - int(y_img[i])) for i in range(min_range)]
    # y_d = [abs(y_template[i] - y_img[i]) for i in range(min_range)]

    # y_d = [(avg_y_template[i] - avg_y_img[i]) for i in range(min_range)] # без абсолютной разницы
    x_d = [i for i in range(len(y_d))]
    plt.title(f'Difference: Max: {max(y_d)} Min: {min(y_d)}')
    plt.plot(x_d, y_d)

    # plot_path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/edge_slice/plots/' + 'plot-' + exp_name
    # путь куда сохраняются плоты
    if destination is None:
        print('Destination path is None')
    destination += exp_name
    #
    plt.savefig(destination, bbox_inches='tight')
    plt.show()
    cv.waitKey(0)
    #plt.close('all')
    print(f'Plot saved: {destination}')

# функция высчитывает среднее значение линии
# template - бунгард / image - изображение / exp_name - название картинки / destination - место сохранения плота
def calc_avg_diff(template, image, exp_name=None, destination=None):
    y, x = image.shape
    y_t, x_t = template.shape

    min_range = min(x, x_t)  # минимальная ширина картинки

    plt.ion()
    plt.figure(figsize=(20, 12))

    avg_y_template = []
    tmp = 0
    for i in range(min_range):  # x_t стояло
        for j in range(y_t):
            tmp += template[j][i]
        avg_y_template.append(tmp // y_t)
        tmp = 0
    x_template = [i for i in range(min_range)]  # x_t стояло
    plt.subplot(3, 1, 1)
    plt.title(f'bungard')
    plt.plot(x_template, avg_y_template)

    avg_y_img = []
    tmp = 0
    for i in range(min_range):     # x стоял
        for j in range(y):
            tmp += image[j][i]
        avg_y_img.append(tmp // y)
        tmp = 0
    x_img = [i for i in range(min_range)]   # x стоял
    plt.subplot(3, 1, 2)
    if exp_name:
        plt.title(f'{exp_name}')
    else:
        plt.title(f'Exp')
    plt.plot(x_img, avg_y_img)


    plt.subplot(3, 1, 3)  # plot of difference between bingard and image
    y_d = [abs(avg_y_template[i] - avg_y_img[i]) for i in range(min_range)]
    #y_d = [(avg_y_template[i] - avg_y_img[i]) for i in range(min_range)] # без абсолютной разницы
    x_d = [i for i in range(len(y_d))]
    plt.title(f'Difference: Max: {max(y_d)} Min: {min(y_d)}')
    plt.plot(x_d, y_d)

    #plot_path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/edge_slice/plots/' + 'plot-' + exp_name
    # путь куда сохраняются плоты
    destination += exp_name

    plt.savefig(destination, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print(f'Plot saved: {destination}')




# функция сохранения изображений графиков разницы
# template - картинка бунгарда / names - список имен изображений /
# source - место, где лежат изображения / destination - место сохранения плота
def save_plots_pictures(template, names, source, destination):
    print(names)
    # путь к месту где лежат изображения
    #path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/'
    for i in range(len(names)):
        img = cv.imread(source + names[i], 0)
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)     # поворот изоображения на 90 градусов противчасовой
        #calc_avg_diff(template, img, names[i], destination)
        calc_diff(template, img, names[i], destination)

    print('Work done!')


if __name__ == '__main__':

    # calc_avg_diff(template_bungard, img, 'jopa')

    # путь к эталону
    template_bungard = cv.imread('C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/bungard.tif', 0)

    pic_names = ['3-3-g.tif', '3-4-g.tif', '3-8-g.tif', '3-11-g.tif',   # названия изображений
             '3-12-g.tif', '3-15-g.tif', '3-16-g.tif', '4-1-1-g.tif',
             '4-1-2-g.tif']

    # путь к месту где лежат изображения
    path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/'

    # путь, куда сохранять плоты
    plot_path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/edge_slice/plots/plot-'

    img = cv.imread('C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/edge_slice_second/3-3.tif', 0)

    calc_diff(template_bungard, img, '1', 'C:/Users/Root/Desktop/template_maker/')

    #save_plots_pictures(template_bungard, pic_names, path, plot_path)


    # while True:
    #     # считываем значения бегунков
    #     h1 = cv.getTrackbarPos('x', 'settings')
    #     s1 = cv.getTrackbarPos('y', 'settings')
    #     cv.imshow('result', img)
    #
    #     print(h1, s1)
    #
    #     if prev_s1 - s1 == 0:
    #         img = cv.imread(path, 0)
    #         cv.line(img, (0, s1), (y_range, s1), (255, 0, 0), 5)
    #
    #     ch = cv.waitKey(5)
    #     if ch == 27:  # exp button
    #         cv.destroyAllWindows()
    #         break
    #
    #     if ch == 32: # space
    #         img = cv.imread(path, 0)
    #         #show_plot(img, x_range, y_range, s1)
    #         show_two_plots(template_bungard, img, x_range, y_range, s1)
    #
    #     if ch == 13: # enter
    #         pass
    #
    #     prev_s1 = s1



# x = [i for i in range(x_range)]
# rotated_image = np.rot90(img)
# line = 500
# y = [rotated_image[line][i] for i in range(x_range)]
#
# print(y)
#
# plt.plot(x, y, label=str(line))
# plt.legend()
# plt.show()



    # x = [i for i in range(4201)]
    #
    # rotated_image = np.rot90(images['2-4.tif'])
    # print(np.shape(rotated_image))
    # y = [rotated_image[500][i][0] for i in range(4201)]
    #
    # #fig, ax = plt.subplots(3, 1)
    # plt.figure(1)
    #
    # plt.subplot(4, 1, 1)
    # plt.plot(x, y, label='500')
    # plt.legend()
    #
    # y = [rotated_image[1500][i][0] for i in range(4201)]
    # plt.subplot(4, 1, 2)
    # plt.plot(x, y, label='1500')
    # plt.legend()
    #
    # y = [rotated_image[2500][i][0] for i in range(4201)]
    # plt.subplot(4, 1, 3)
    # plt.plot(x, y, label='2500')
    # plt.legend()
    #
    # plt.subplot(4, 1, 4)
    # t = np.polyfit(x, y, 40)    # интерполяция полиномом 40 степени
    # f = np.poly1d(t)
    # x_l = np.linspace(0, 4000, 100)
    # #np.linspace()
    # print(f)
    # plt.plot(x_l, f(x_l), label='2500')
    # plt.legend()
    #
    # from scipy import interpolate
    # plt.figure(2)
    # #temp = interpolate.interp1d(x, y)
    # #ynew = temp(x)
    # x_l = np.linspace(0, 4000, 100)
    # temp = interpolate.splrep(x, y, s=0)
    # ynew = interpolate.splev(x_l, temp, der=0)
    # plt.plot(x_l, ynew)
    #
    #
    # plt.figure(3)
    # plt.imshow(rotated_image)
    # #  plt.plot(images['2-1.tif'])
    # plt.show()