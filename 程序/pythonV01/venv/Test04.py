import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# 加载实际数据;
digits=datasets.load_digits()

# digits应具有2个不同的数据域：data域包含了图像数据，target包含了图像的标签，
# 可以用来展现数据的其他域的信息;
# print(digits)
print(digits.data.shape)
print(digits.images.shape)
#第1维对应的都是数据集合中的图像的数量。然而，data中所有像素都在一个大的向量中排列，而images保留了各个像素8x8的空间排列
# (1797, 64)
# (1797, 8, 8)

img=digits.images[0, :, :]#从1797个元素的数组中获取其第一行数据，这行数据对应的是8x8=64个像素
# # 元素的标签的显示方式;
# plt.imshow(img,cmap="gray")
#
# # plt数据的显示方式;
# plt.show()

# 默认情况下，Matplotlib使用MATLAB默认的颜色映射为jet。然而，灰度图像的情况下，gray颜色映射更加有效;

for image_index in range(10):#看成从0开始直至10以前的数字;
    subplot_index=image_index+1
    # subplot函数需要制定行数、列数以及当前的子绘图索引（从1开始计算）
    plt.subplot(2,5,subplot_index)
    plt.imshow(digits.images[image_index, :, :],cmap="gray")

plt.show()