import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 读取 .npy 文件的数据
data = np.load(r"C:\Users\liushuo\Desktop\github代码\SMAP&MSL\SMAP&MSL\train\C-2.npy")

# 打印读取到的数据
data = pd.DataFrame(data)
print(data)
plt.plot(data.iloc[:,54])
# plt.subplot(221)
# plt.plot(data.iloc[11000:12000,0])
#
# plt.subplot(222)
# plt.plot(data.iloc[21000:22000,1])
#
# plt.subplot(223)
# plt.plot(data.iloc[21000:23000,5])
#
plt.show()
