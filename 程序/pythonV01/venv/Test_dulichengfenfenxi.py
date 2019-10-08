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
ica=decomposition.FastICA()
X2=ica.fit_transform(X)
plt.plot(X2[:,0],X2[:,1],'o')
plt.xlabel("first independent component")
plt.ylabel("second independent component")
plt.axis([-0.2,0.2,-0.2,0.2])
plt.show()