import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import cv2 as cv


pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/cropped/3-8.tif'
#pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/bungard.tif'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/scan0001.tif'
# image_names = [str(i) + '.png' for i in range(6)]  # названия картинок к обработке
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/cropped.tif'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/cropped.tif'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/cropped.tif'
#pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/cropped/3-8.tif'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/scan0002.tif'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/29.12.2022/cropped/3-11.tif'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/3-3-29.tif'
pic_path = 'C:/Users/Root/Desktop/template_maker/res.tif'

class Crop:

    def __init__(self):
        # self.img = cv.imread(path, 0)
        #self.img = cv.imread(path, cv.COLOR_BGR2GRAY)
        # self.img = cv.imread(path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        # self.img = cv.imread(path, -1)

        self.img_blur = None    # for cutted and blured pic
        self.path = None    # path to load picture
        self.x1 = 0     # first and second point to cut
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.counter = 0    # counter for clicking
        self.points = None  # may be delete just in case of not clearing points on plot
        self.lock_coords_change = False     # flag for locking update of cut points

        self.curr_image_index = 0
        self.image_names = [str(i) + '.png' for i in range(6)]

    def show_picture(self, path):
        #self.img = cv.imread(path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        self.img = cv.imread(path, -1)       # считывает изображение в исходном виде

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)
        self.ax.set_title('picture')
        self.fig.set_figwidth(12)  # ширина и
        self.fig.set_figheight(8)  # высота "Figure"
        plt.subplots_adjust(bottom=0.15)  # приподнимает график отнижней грани

        mng = plt.get_current_fig_manager()  # блокирует изменение размера окна
        mng.window.resizable(False, False)

        ax_button_cut = plt.axes([0.25, 0.08, 0.08, 0.05])  # отрисовка кнопки Cut
        button_cut = Button(ax_button_cut, 'Cut', color='white', hovercolor='grey')

        ax_button_next = plt.axes([0.35, 0.08, 0.08, 0.05])  # отрисовка кнопки Next
        button_next = Button(ax_button_next, 'Next', color='white', hovercolor='grey')

        button_cut.on_clicked(self.button_cut_clicked)  # проверка нажатия кнопки Cut
        button_next.on_clicked(self.button_next_clicked)  # проверка нажатия кнопки Next

        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()

    def close_all(self):
        plt.close()

    def increment(self):    # проверка количеств нажатий кнопки
        self.counter += 1
        if self.counter >= 2:
            #plt.close('all')
            self.counter = 0

            #self.fig.show()
            # self.img = self.img[self.y1:self.y2, self.x1:self.x2]
            if self.lock_coords_change == False:
                self.x1, self.x2, self.y1, self.y2 = 0, 0, 0, 0

    def crop_points(self, x, y):
        if self.counter == 1:  # если второй клик сделан, то ставим координаты второй точки
            self.x2, self.y2 = x, y
        if self.counter == 0:       # если первый клик сделан, то координаты первой точки
            self.x1, self.y1 = x, y

        #print(self.x1, self.y1, self.x2, self.y2)

    def draw_point(self, x, y): # отрисовка точек на plot
        self.points = self.ax.scatter(x, y, c='deeppink', s=8.9)  # цвет точек и размер
        #self.fig.show()
        plt.show()

    def button(self):  # должная была отрисовывать кнопку, но через вызов метода не отрисовывает
        ax_button = plt.axes([0.25, 0.08, 0.08, 0.05])
        grid_button = Button(ax_button, 'Cut', color='white', hovercolor='grey')

    def save_cropped_pic(self):
        self.img_blur = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img_blur = self.img_blur[self.y1:self.y2, self.x1:self.x2]
        # self.img_blur = self.img[0:6476, 0:3898]
        self.img_blur = cv.GaussianBlur(self.img_blur, (25, 25), 5)     # применение фильтра гаусса
        cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/cropped.tif', self.img_blur, ((int(cv.IMWRITE_TIFF_RESUNIT), 2,
                                                                  int(cv.IMWRITE_TIFF_COMPRESSION), 1,
                                                                  int(cv.IMWRITE_TIFF_XDPI), 600,
                                                                  int(cv.IMWRITE_TIFF_YDPI), 600)))
        # cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/29.12.2022/cropped.tif', self.img_blur)

    def button_cut_clicked(self, val):
        print('button cut clicked, cutted picture saved!')
        self.save_cropped_pic()
        self.lock_coords_change = False

    def button_next_clicked(self, val):
        self.curr_image_index += 1
        print(f'button next clicked: {self.curr_image_index}')

    def onclick(self, event):
        print(f'x: {round(event.xdata)}    | y: {round(event.ydata)}')
        # self.increment()
        print(f'counter of clicks: {self.counter}')
        if self.counter >= 1:       # засчита от переключения
            self.lock_coords_change = True

        if round(event.xdata) not in [0, 1] and round(event.ydata) not in [0, 1]:   # проверка, чтобы не засчитывал координаты при кликах на кнопки
            self.crop_points(round(event.xdata), round(event.ydata))
            self.draw_point(round(event.xdata), round(event.ydata))
            self.increment()

    def __del__(self):
        print('deleted')

if __name__ == '__main__':

    work = Crop()

    work.show_picture(pic_path)

    # print(image_names)
    # while True:
    #     if work.curr_image_index == 2:
    #         work.close_all()
    #         work.show_picture(pic_path + image_names[work.curr_image_index])