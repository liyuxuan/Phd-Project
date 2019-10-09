import cv2
import matplotlib.pyplot as plt

img_bgr=cv2.imread("figures/pangde.png")

img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
# 设置角点检测的像素邻域大小(blocksize),边缘检测的孔径参数(ksize),以及所谓点的Harris检测器的自由参数(k)

corners=cv2.cornerHarris(img_gray,2,3,0.04)
plt.imshow(corners,cmap="gray")
plt.show()