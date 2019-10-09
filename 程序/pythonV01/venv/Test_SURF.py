import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr=cv2.imread("figures/pangde.png")
img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

surf=cv2.xfeatures2d.SURF_create()
kp=surf.detect(img_bgr)
img_kp=np.zeros_like(img_bgr)

img_kp=cv2.drawKeypoints(img_rgb,kp,img_kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp,des=surf.compute(img_bgr,kp)
print(des.shape)

plt.imshow(img_kp)
plt.show()