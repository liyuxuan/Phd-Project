import numpy as np
import matplotlib.pyplot as plt
import cv2

mean=[20,20]
cov=[[5,0],[25,25]]
x,y=np.random.multivariate_normal(mean,cov,1000).T
X=np.vstack((x,y)).T

# mu,eig=cv2.PCACompute(X,np.array([]))

# X2=cv2.PCAProject(X,mu,eig)
from sklearn import decomposition

nmf=decomposition.NMF()
X2=nmf.fit_transform(X)
plt.plot(X2[:,0],X2[:,1],'o')
plt.xlabel("first non-negative component")
plt.ylabel("second non-negative component")
plt.axis([-5,15,-5,15])
plt.show()