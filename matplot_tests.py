#import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, CheckButtons

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.subplots_adjust(bottom=0.3)
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 10])
# counter = 0
#
# def onclick(event):
#     print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           (event.button, event.x, event.y, event.xdata, event.ydata))
#     plt.plot(event.xdata, event.ydata, '+')
#     #fig.canvas.draw()
#     if counter > 5:
#         del ax.collections[:]
#     fig.canvas.draw_idle()
#
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
#
#
# plt.show()

import matplotlib.pyplot as plt
import cv2 as cv
# def onclick(event):
#     paths.remove()
#     plt.show()
#
#
# fig, ax = plt.subplots()
# paths = ax.scatter([3,2,1,4], [2,3,4,0])
#
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()

#pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/0.tiff'
pic_path = 'C:/Users/Root/Documents/MEGAsync/diplom/02.04.2023/new_scans/3-3-g.tif'
# img = cv.imread(pic_path, -1)
img = cv.imread(pic_path)



#
# fig, ax = plt.subplots()
# img = cv.cvtColor(img, 2)
#
# print(img.shape)
# img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
# print(img.shape)
# cv.imshow('result', img)
# cv.waitKey(0)
#
# ax.imshow(img)
# ax.set_title('picture')
# fig.set_figwidth(10)  # ширина и
# fig.set_figheight(6)  # высота "Figure"
# plt.show()
# plt.close('all')

def foo(*args):
    for n in args:
        print(n)


# foo(1, 3, 5, 5, 6, 7)
# foo(*[1, 3, 5, 5, 6])

def goo(lst):
    # for i in range(len(lst)):
    #     lst[i] += i
    lst = lst[::-1].copy()
    print(lst)

l = [1, 2, 3, 4]
print(l)
goo(l)
print(l)


#paths.remove() # to just remove the scatter plot and keep the limits

