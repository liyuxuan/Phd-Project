import cv2
import matplotlib.pyplot as plt

img_bgr=cv2.imread("figures/pangde.png")
# #  matplotlib使用RGB图像会出现与原色极大不同的色彩图像，
# # 希望输出为RGB色彩，则转换为RGB图像格式，使用cv2.cvtColor重新排列色彩通道；
# img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
#
# plt.figure(figsize=(12,6))
#
# plt.subplot(121)
# plt.imshow(img_bgr)
# plt.subplot(122)
###############################################################
###############################################################
# HSV色彩空间表示
# img_hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
# img_hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HLS)
# img_hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2LAB)
img_hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YUV)

plt.imshow(img_hsv)
plt.show()