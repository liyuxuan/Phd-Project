import numpy as np
mean=[20,20]
cov=[[5,0],[25,25]]
x,y=np.random.multivariate_normal(mean,cov,1000).T
#
# # print(x,y)
#
# #使用matlab绘制出相应的图像;
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#
# plt.plot(x,y,'o',zorder=1)
# plt.axis([0,40,0,40])
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
#
# plt.show()

X=np.vstack((x,y)).T

# 在特征矩阵X上计算PCA。指定空数组np.array([])
import cv2
mu,eig=cv2.PCACompute(X,np.array([]))
print(eig)
###################################################################
#函数包括两个值：投影前减去的平均值（mean）和协方差矩阵的特征向量（eig），这些特征指向PCA认为最有信息性的方向
#协方差是不具有对称性的;
# plt.plot(x,y,'o',zorder=1)
# plt.quiver(mean[0],mean[1],eig[:,0],eig[:,1],zorder=3,scale=0.2,units='xy')
# plt.text(mean[0]+5*eig[0,0],mean[1]+5*eig[0,1],'u1',zorder=5,fontsize=16,bbox=dict(facecolor='white',alpha=0.6))
#
# plt.text(mean[0]+7*eig[1,0],mean[1]+4*eig[1,1],'u2',zorder=5,fontsize=16,bbox=dict(facecolor='white',alpha=0.6))

# plt.axis([0,40,0,40])
# plt.xlabel("feature 1")
# plt.ylabel("feature 2")
# plt.show()
##################################################################
##################################################################
#第一个特征向量(u1)指向是数据分布的最大方向,称为第一主成分;u2代表第2主成分，表示的是观察数据的第二个主要的的方差方向;由此可知x坐标与y坐标无法进行有效分析，选择u1,u2轴更有意义;
X2=cv2.PCAProject(X,mu,eig)
#进行处理之后数据应该完成了旋转;
plt.plot(X2[:,0],X2[:,1],'o')
plt.xlabel("first principal component")
plt.ylabel("second principal component")
plt.axis([-20,20,-10,10])
plt.show()