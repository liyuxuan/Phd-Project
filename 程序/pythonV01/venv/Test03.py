import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x=np.linspace(0,10,100)
p=plt.plot(x,np.sin(x))
# 从脚本中进行运行，需要进行调用,在程序的末尾调用这个函数;
print(p)

# 图像的风格;
p1=plt.style.available
print(p1)
# 对图像进行样式的分析;
plt.style.use("seaborn-dark")

# 进行plt中的图像的保存;
plt.savefig("figures/02.04-sine.png")

plt.show()


