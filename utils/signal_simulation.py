from arw_simulation import  signal1, signal2, step_position
import matplotlib.pyplot as plt
import os
import numpy as np

# 创建图形
fig = plt.figure(figsize=(12, 8))

# 第一个子图
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(signal1)) / 100,signal1 * 3600)
plt.xlabel("Time(s)")
plt.ylabel('FOG output(°/h)')
pos1 = plt.gca().get_position()  # 获取第一个子图的位置信息

# 第二个子图
plt.subplot(1, 2, 2)
plt.plot(np.arange(len(signal1)) / 100,signal2 * 3600)
plt.xlabel("Time(s)")
plt.ylabel('FOG output(°/h)')
pos2 = plt.gca().get_position()  # 获取第二个子图的位置信息

# 添加标注
fig.text(pos1.x0 + pos1.width/2, pos1.y0 - 0.08, '(a)',
         fontsize=12, ha='center')
fig.text(pos2.x0 + pos2.width/2, pos2.y0 - 0.08, '(b)',
         fontsize=12, ha='center')

# ============ 新增：保存图片部分 ============
# 1. 构建保存路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
parent_dir = os.path.dirname(current_dir)  # 上一级目录
pic_dir = os.path.join(parent_dir, 'pic')  # 上一级目录中的pic文件夹

# 2. 如果pic文件夹不存在，创建它
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

# 3. 保存为PDF文件
pdf_path = os.path.join(pic_dir, 'origin_signal.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300)
print(f"图片已保存到: {pdf_path}")

plt.show()