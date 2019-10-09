import cv2
import matplotlib.pyplot as plt

img_bgr=cv2.imread("figures/pangde.png")

img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

sift=cv2.xfeatures2d.SIFT_create()

#SIFT算法分成两步骤:
# 1.检测：识别图像中的兴趣点;
# 2.计算：计算每个关键点真实的特征值;
kp=sift.detect(img_bgr)
# drawKeypoints函数提供了一个对检测到的关键点进行可视化非常好的方式.
# 通过传入一个选项标记,cv2.DRAW_MATHCES_FLAGS_RICH_KEYPOINTS函数就会在每个关键点上绘制圆圈，
# 圆圈的大小：表示关键点的重要性。
# 圆圈的径向线：表示关键点的方向。
import numpy as np

img_kp=np.zeros_like(img_bgr)
img_kp=cv2.drawKeypoints(img_rgb,kp,img_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#描述关键点的实际信息;
kp,des=sift.compute(img_bgr,kp)
#通过特征描述符应该有128个特征值点,用于找到每一个关键点。
out=des.shape
print(out)
##################################################################################
##################################################################################
#2.检测关键点并且计算特征描述符
kp2,des2=sift.detectAndCompute(img_bgr,None)
#使用Numpy,通过确保des中的每个值都近似等于des2中的值，可以确信两种方式一样;
# 比较两种方式是否一样的结果;
out2=np.allclose(des,des2)
print(out2)

plt.imshow(img_kp)

plt.show()