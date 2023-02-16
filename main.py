import math
from decimal import *
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cv2 as cv
import random
from scipy.stats import moment

from constants import *     # импорт констатнт


images = {}
histograms = {}
# parallel_experiments = {'2-1.tif': 6, '2-2.tif': 6, '2-3.tif': 4, '2-4.tif': 6, '2-5.tif': 6, '2-6.tif': 6,
#                         '2-7.tif': 6, '2-8.tif': 6, '2-9.tif': 6, '2-10.tif':6, '2-11.tif': 6, '2-12.tif': 4,
#                         '2-13.tif': 6, '2-14.tif': 6, '2-15.tif': 6, '2-16.tif': 6
#                         }

parallel_experiments = {'3-3-g.tif': 6, '3-4-g.tif': 6, '3-8-g.tif': 4, '3-11-g.tif': 6, '3-12-g.tif': 6,
                        '3-15-g.tif': 6, '3-16-g.tif': 6, '4-1-1-g.tif': 6, '4-1-2-g.tif': 6
                        }

# params = {'2-1.tif': 'T = 40 s, H = 0 mm', '2-2.tif': 'T = 50 s, H = 0 mm',
#           '2-3.tif': 'T = 60 s, H = 0 mm', '2-4.tif': 'T = 70 s, H = 0 mm',
#           '2-5.tif': 'T = 40 s, H = 20 mm', '2-6.tif': 'T = 50 s, H = 20 mm',
#           '2-7.tif': 'T = 60 s, H = 20 mm', '2-8.tif': 'T = 70 s, H = 20 mm',
#           '2-9.tif': 'T = 40 s, H = 40 mm', '2-10.tif': 'T = 50 s, H = 40 mm',
#           '2-11.tif': 'T = 60 s, H = 40 mm', '2-12.tif': 'T = 70 s, H = 40 mm',
#           '2-13.tif': 'T = 40 s, H = 60 mm', '2-14.tif': 'T = 50 s, H = 60 mm',
#           '2-15.tif': 'T = 60 s, H = 60 mm', '2-16.tif': 'T = 70 s, H = 60 mm',
#           '2-17.tif': 'T = 14 s, H = 0 mm'
#           }

params = {'3-3-g.tif': 'T = 30 s, H = 0 mm', '3-4-g.tif': 'T = 40 s, H = 0 mm',
          '3-8-g.tif': 'T = 40 s, H = 20 mm', '3-11-g.tif': 'T = 30 s, H = 30 mm',
          '3-12-g.tif': 'T = 40 s, H = 30 mm', '3-15-g.tif': 'T = 30 s, H = 40 mm',
          '3-16-g.tif': 'T = 40 s, H = 40 mm', '4-1-1-g.tif': 'T = 50 s, H = 100 mm',
          '4-1-2-g.tif': 'T = 70 s, H = 100 mm'}

# parallel_experiments = {'3-1.tif': 6, '3-2.tif': 6, '3-3.tif': 6, '3-4.tif': 6, '3-5.tif': 6, '3-6.tif': 6,
#                         '3-7.tif': 6, '3-8.tif': 6, '3-9.tif': 6, '3-10.tif': 6, '3-11.tif': 6, '3-12.tif': 6,
#                         '3-13.tif': 6, '3-14.tif': 6, '3-15.tif': 6, '3-16.tif': 6, '3-17.tif': 6
#                         }
#
# params = {'3-1.tif': 'T = 10 s, H = 0 mm', '3-2.tif': 'T = 20 s, H = 0 mm',
#           '3-3.tif': 'T = 30 s, H = 0 mm', '3-4.tif': 'T = 40 s, H = 0 mm',
#           '3-5.tif': 'T = 10 s, H = 20 mm', '3-6.tif': 'T = 20 s, H = 20 mm',
#           '3-7.tif': 'T = 30 s, H = 20 mm', '3-8.tif': 'T = 40 s, H = 20 mm',
#           '3-9.tif': 'T = 10 s, H = 30 mm', '3-10.tif': 'T = 20 s, H = 30 mm',
#           '3-11.tif': 'T = 30 s, H = 30 mm', '3-12.tif': 'T = 40 s, H = 30 mm',
#           '3-13.tif': 'T = 10 s, H = 40 mm', '3-14.tif': 'T = 20 s, H = 40 mm',
#           '3-15.tif': 'T = 30 s, H = 40 mm', '3-16.tif': 'T = 40 s, H = 40 mm',
#           '3-17.tif': 'T = 14 s, H = 0 mm'
#           }

# path = 'C:/Users/Root/Desktop/Exposure_experiment/second/gs/GS-3/'
# path = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/'
path = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/first exp cropped by python/'
path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/'


# переписать в универсальную функцию через *args
def calc_hist():
    #for image_number in range(1, 17):
    for k in parallel_experiments.keys():
        #image_name = f"2-{image_number}.tif"
        image_name = k
        #image_name.append(f"2-{image_number}.tif")
        images.setdefault(image_name, cv.imread(path + image_name))
        histograms.setdefault(image_name, cv.calcHist([images[image_name]], [0], None, [256], [0, 256]))
        sum_quantity = 0
        for i in histograms[image_name]:
            i[0] = i[0] / parallel_experiments[image_name]
            sum_quantity = sum_quantity + i[0]
        for i in histograms[image_name]:
            i[0] = i[0] * 100 / sum_quantity


# фнкция вычилсения одной гистограммы
# image - изображение / d - делитель
def calc_single_hist(image, d=6):
    hist = cv.calcHist(image, [0], None, [256], [0, 256])
    sum_quantity = 0
    for i in hist:
        i[0] = i[0] / d
        sum_quantity = sum_quantity + i[0]
    for i in hist:
        i[0] = i[0] * 100 / sum_quantity
    return hist




# расчет мат ожидания
def calc_mat_expectation(lst):  # на вход подается массив значений
    res = 0
    my_list = lst.flatten()  # преобразовние
    sum_of_pixels = sum(my_list)    # считаем общее кол-во значений тонов серого
    for pixel_num in range(256):    # проходим по каждому тону
        p_i = my_list[pixel_num] / sum_of_pixels    # вычисление вероятности данного тона
        res += p_i * pixel_num      # мат ожидание для i-го тона
    return res


def calc_dispersion(lst, mat):
    # res = 0
    my_list = lst.flatten()  # преобразовние

    # sum_of_pixels = sum(my_list)  # считаем общее кол-во значений тонов серого
    # for pixel_num in range(256):    # проходим по каждому тону
    #     res += (pixel_num - mat) ** 2    # вычисление вероятности данного тона
    # return res // 256
    my_list = list(map(lambda x: x ** 2, my_list))
    print(*my_list)
    print(calc_mat_expectation(np.array(my_list)))
    return calc_mat_expectation(np.array(my_list)) - mat ** 2


def range_of_hist(lst):     # расчет границ гистограммы очень грубым методом
    res = []
    for i in range(256):    # расчет левой границы
        if 1 - lst[i] <= 0.9:   # если разница между высотой и 1 меньше 0.9
            res.append(i)
            break

    for i in range(255, 0, -1):      # расчет правой границы
        if 1 - lst[i] <= 0.9:
            res.append(i)
            break
    return res


def calc_size(dic, img):    # расчет "площади" методом Монте-Карло (работает вроде нормально)
    res = {}

    for i in range(4):      #  для четырех гистограмм
        width = range_of_hist(dic[img[i]])      # ширина
        n = 10 ** 6     # кол-во итераций
        k = 0
        max_h = max(dic[img[i]])        # высота прямоугольника = максимальному значению гистграммы
        s0 = max_h * (width[1] - width[0])      # площадь прямоугольника
        print(f'widfht = {width[1] - width[0]}, {img[i]}')
        for _ in range(n):
            x = random.randint(width[0], width[1])   # случайная точка в пределах икса
            y = random.uniform(0, float(max_h))     # случайная высота
            if y < dic[img[i]][x][0]:  # если случайная точка находится под графиком гистограммы
                k += 1  # увеличиваем счетчик попавших точек
        res[img[i]] = (k / n) * s0      # добавляем результат в словарь (имя_картинки: площадь)

    return res


# рисует графики для четырех высот с мат ожиданием и границами
def show_plot_with_const_time(dic, img):   # на вход передаем словарь файл/гистограма и список с 4-мя файлами
    #keys = list(matrix_const_height.keys())
    plt.figure(figsize=(8, 8))

    for i in range(4):
        plt.subplot(2, 2, i + 1)

        plt.title(f'{img[i]} {params[img[i]]}')
        plt.ylabel('value')
        plt.xlabel('tone')
        plt.plot(dic[img[i]], label='distribution')

        mat = Decimal(calc_mat_expectation(dic[img[i]]))    # вычисление мат ожидания для i-го рисунка
        plt.axvline(x=mat, ymin=0.05, ymax=0.95, color='purple', ls='--', label=f'M[x]={mat.quantize(Decimal("1.000"))}')

        plt.axvline(x=range_of_hist(dic[img[i]])[0], ymin=0.05, ymax=0.55, color='red', ls='--')
        plt.axvline(x=range_of_hist(dic[img[i]])[1], ymin=0.05, ymax=0.55, color='red', ls='--')

       # moda = calc_mode(dic[matrix_const_time[time][i]])
       # plt.plot(10, 10, label=f'Moda = {moda}')
        #plt.legend(['distribution', f'M[x]={mat.quantize(Decimal("1.000"))}', 'moda'], loc=2)

        plt.legend(loc=2, shadow=True)
        #start = ''.join(dic[images[i]].flatten()).lstrip('0')
        plt.xlim([mat-15, mat+25])  # лимит по ширине

    plt.show()


# вывод 1 графика
def show_one_plot(dic, img):    # на вход подать словарь с гистограммами и название картинки
    plt.figure(figsize=(8, 8))
    plt.title(f'current image: {img}')
    plt.ylabel('value')
    plt.xlabel('tone')
    plt.plot(dic[img], label='distribution')
    mat = Decimal(calc_mat_expectation(dic[img]))  # вычисление мат ожидания для i-го рисунка
    plt.axvline(x=mat, ymin=0.05, ymax=0.95, color='purple', ls='--', label=f'M[x]={mat.quantize(Decimal("1.000"))}')

    plt.axvline(x=range_of_hist(dic[img])[0], ymin=0.05, ymax=0.55, color='red', ls='--')
    plt.axvline(x=range_of_hist(dic[img])[1], ymin=0.05, ymax=0.55, color='red', ls='--')

    plt.legend(loc=2, shadow=True)
    plt.xlim([mat - 15, mat + 25])  # лимит по ширине
    plt.show()

# функция строит график изменения площади от времени экспонирования или высоты проставки
def show_plot_with_size_changing(square, name='distribution'):
    #mm = [0, 20, 40, 60]
    mm = [40, 50, 60, 70]
    square = list(square)
    plt.title(f'S(t)')
    plt.ylabel('square ')
    plt.xlabel('s')
    plt.plot(mm, square, label=name)

    for i in range(4):
        plt.annotate(*square[i], (mm[i] + 0.1, square[i] + 0.1))

    #plt.show()


# ааа мб это выисление гистограммы по правилу Стреджесса, что для оптимального изображение гистограммы можно высчитать определенное число разбиений
def calc_hist_stredges(dic):        # я не помню что это, но судя по всему она должна была 8 графиков рисовать с мат ожиданиями
    plt.figure(figsize=(8, 8))      # никаких 8, а 1, но с столбчатым распределением?

    #plt.title(f'{img[i]} {params[img[i]]}')
    plt.ylabel('value')
    plt.xlabel('tone')
    plt.plot(dic, label='distribution')

    mat = Decimal(calc_mat_expectation(dic))  # вычисление мат ожидания для i-го рисунка
    plt.axvline(x=mat, ymin=0.05, ymax=0.95, color='purple', ls='--',
                label=f'M[x]={mat.quantize(Decimal("1.000"))}')

    # plt.axvline(x=range_of_hist(dic[img[i]])[0], ymin=0.05, ymax=0.55, color='red', ls='--')
    # plt.axvline(x=range_of_hist(dic[img[i]])[1], ymin=0.05, ymax=0.55, color='red', ls='--')


    plt.legend(loc=2, shadow=True)
    # start = ''.join(dic[images[i]].flatten()).lstrip('0')
    plt.xlim([mat - 15, mat + 15])  # лимит по ширине

    plt.show()


def save_math_expectations():
    with open('math_expectations.txt', 'a', encoding='utf-8') as file:
        for name in height_60_mm:
            mat = Decimal(calc_mat_expectation(histograms[name]))
            print(name, mat.quantize(Decimal("1.000")), file=file)


def save_distribution():
    pass


if __name__ == '__main__':

    calc_hist()

    #show_plot_with_const_time(histograms, time_70s)
    show_one_plot(histograms, '3-3-g.tif')
    # for k in parallel_experiments.keys():
    #
    #     show_one_plot(histograms, k)


   # 4 графика отдельными
   #  show_plot_with_size_changing(calc_size(histograms, time_40s).values(), time[0])
   #  plt.show()
   #  show_plot_with_size_changing(calc_size(histograms, time_50s).values(), time[1])
   #  plt.show()
   #  show_plot_with_size_changing(calc_size(histograms, time_60s).values(), time[2])
   #  plt.show()
   #  show_plot_with_size_changing(calc_size(histograms, time_70s).values(), time[3])
   #  plt.show()


    # 4 графика на одном
    # строит 4 графика изменения ширина от высоты проставки или времени экспонирования
    # for j in range(4):
    #     square = calc_size(histograms, matrix_2[j])
    #     show_plot_with_size_changing(square.values(), time_2[j])
    #     plt.legend(loc=3, shadow=True)
    # plt.show()















    # судя по всему тут вбиты данные ширин гистограмм и просто строятся графики
    # x = [0, 20, 40, 60] # изменение ширины гистограммы при росте времени экспонирования
    # y = [17, 11, 14, 19]
    # plt.plot(x, y, label='time_40_s')
    #
    # x = [0, 20, 40, 60]  # изменение ширины гистограммы при росте времени экспонирования
    # y = [17, 11, 12, 17]
    # plt.plot(x, y, label='time_50_s')
    #
    # x = [0, 20, 40, 60]  # изменение ширины гистограммы при росте времени экспонирования
    # y = [13, 11, 9, 13]
    # plt.plot(x, y, label='time_60_s')
    #
    # x = [0, 20, 40, 60]  # изменение ширины гистограммы при росте времени экспонирования
    # y = [14, 7, 13, 9]
    # plt.plot(x, y, label='time_70_s')
    #
    # plt.legend(loc=3, shadow=True)
    # plt.show()





