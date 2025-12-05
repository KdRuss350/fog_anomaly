import os
import matplotlib.gridspec as gridspec
from arw_simulation import arw1
from arw_simulation import arw2
from arw_simulation import signal1
from arw_simulation import signal2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'   # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False        # 允许负号显示

# ------------------------------------------------------------
# 动态 Allan 方差计算（窗口差分法）
# ------------------------------------------------------------
def compute_dynamic_avar(signal, window_size = 200, step_size = 4, tau_values=None, dt=1):
    if tau_values is None:
        tau_values = np.linspace(1, 50, num=50)
    davar_matrix = []
    time_centers = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        w = signal[start:start + window_size]
        time_centers.append((start + window_size // 2) * dt)  # 改成时间

        avars = []
        for tau in tau_values:
            m = int(round(tau / dt))
            if m < 2:
                avars.append(np.nan)
                continue

            k = len(w) // m
            if k < 2:
                avars.append(np.nan)
                continue

            y_bar = np.mean(w[:k*m].reshape(k, m), axis=1)
            diff = y_bar[1:] - y_bar[:-1]
            avar = 0.5 * np.mean(diff ** 2)
            avars.append(avar)
        davar_matrix.append(avars)

    return np.array(time_centers), np.array(tau_values), np.array(davar_matrix)


# ------------------------------------------------------------
# 绘制 D-AVAR 3D 图
# ------------------------------------------------------------
def plot_davar_3d(time_centers, tau_array, davar_matrix, ax):
    """
    绘制 3D DAVAR 曲面，Z 轴 log 刻度，去掉颜色条，使用鲜明的红绿蓝渐变
    """

    # ----- meshgrid -----
    T, Tau = np.meshgrid(time_centers, tau_array, indexing='ij')
    adev = np.sqrt(davar_matrix)
    Z = adev

    # ----- log transform -----
    Tau_log = np.log10(Tau)
    Z_log = np.log10(Z)

    # ----- 3D figure -----
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
    ax.xaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.5)  # RGBA, 透明度可调
    ax.yaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.5)
    ax.zaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.5)

    # ----- plot surface -----
    surf = ax.plot_surface(
        Tau_log, T, Z_log,
        cmap='jet',       # 红绿蓝渐变
        linewidth=0,
        antialiased=False,
        alpha=0.9
    )

    # ----- τ axis (log) -----
    tau_ticks_log = np.log10(tau_array)
    tau_labels = [""] * len(tau_array)

    # 只显示 log10(t)=0 (1) 和 log10(t)=1 (10)
    for i, t_log in enumerate(tau_ticks_log):
        if np.isclose(t_log, -1) or np.isclose(t_log, -2):
            tau_labels[i] = f"$10^{{{int(t_log)}}}$"

    ax.set_xticks(tau_ticks_log)
    ax.set_xticklabels(tau_labels, fontdict={"family": "Times New Roman", "size": 10})
    ax.set_xlabel("τ", fontdict={"family": "Times New Roman", "size": 16})

    # ----- T axis -----
    # 获取当前刻度
    yticks = ax.get_yticks()
    # 设置刻度标签，同时修改字体
    ax.set_yticklabels([f"${t:g}$" for t in yticks], fontsize=10)
    ax.invert_yaxis()  # 翻转 Y 轴方向
    ax.set_ylabel("t", fontdict={"family": "Times New Roman", "size": 16})
    # ----- Z axis (log) -----
    z_ticks = [1, 0.1, 0.01]
    z_ticks_log = np.log10(z_ticks)
    ax.set_zlabel("σ(t, τ)", fontdict={"family": "Times New Roman", "size": 16})
    ax.set_zticks(z_ticks_log)
    ax.set_zticklabels(
        [r"$10^{0}$", r"$10^{-1}$", r"$10^{-2}$"],
        fontdict={"family": "Times New Roman", "size": 10}  # 设置字体和大小
    )

# ------------------------------------------------------------
# D-AVAR for signal1
# ------------------------------------------------------------
t1, tau1, davar1 = compute_dynamic_avar(signal1 * 3600, window_size=1000, step_size=5)

# plot_davar_3d(t1/100, tau1/100, davar1)
# ------------------------------------------------------------
# D-AVAR for signal2
# ------------------------------------------------------------
t2, tau2, davar2 = compute_dynamic_avar(signal2 * 3600, window_size=1000, step_size=5)
# plot_davar_3d(t2/100, tau2/100, davar2)

fig1 = plt.figure(figsize=(16, 8))
# 使用 gridspec 设置宽度比例
gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])  # 左:右 = 1:1
# ----- subplot 1: 3D DAVAR -----
ax1 = fig1.add_subplot(gs[0], projection='3d')
plot_davar_3d(t1/100, tau1/100, davar1, ax=ax1)
# ----- subplot 2: ARW 曲线图 -----
ax2 = fig1.add_subplot(gs[1])   # 普通 2D subplot
ax2.plot(np.arange(len(arw1)) / 100, arw1)
ax2.set_xlabel("Time")
ax2.set_ylabel("N")
plt.subplots_adjust(wspace=0.3)

# 获取子图位置
pos1 = ax1.get_position()
pos2 = ax2.get_position()

fig1.text(pos1.x0 + pos1.width/2, pos1.y0 - 0.10, '(a)',
         fontsize=16,  ha='center')
fig1.text(pos2.x0 + pos2.width/2, pos2.y0 - 0.10, '(b)',
         fontsize=16,  ha='center')


fig2 = plt.figure(figsize=(16, 8))
# 使用 gridspec 设置宽度比例
gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])  # 左:右 = 1:1
# ----- subplot 1: 3D DAVAR -----
ax3 = fig2.add_subplot(gs[0], projection='3d')
plot_davar_3d(t2/100, tau2/100, davar2, ax=ax3)
# ----- subplot 2: ARW 曲线图 -----
ax4 = fig2.add_subplot(gs[1])   # 普通 2D subplot
ax4.plot(np.arange(len(arw2)) / 100, arw2)
ax4.set_xlabel("Time")
ax4.set_ylabel("N")
plt.subplots_adjust(wspace=0.3)

# 获取子图位置
pos3 = ax3.get_position()
pos4 = ax4.get_position()

fig2.text(pos3.x0 + pos3.width/2, pos3.y0 - 0.10, '(a)',
         fontsize=16,  ha='center')
fig2.text(pos4.x0 + pos4.width/2, pos4.y0 - 0.10, '(b)',
         fontsize=16,  ha='center')

# ============ 新增：保存图片部分 ============
# 1. 构建保存路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
parent_dir = os.path.dirname(current_dir)  # 上一级目录
pic_dir = os.path.join(parent_dir, 'pic')  # 上一级目录中的pic文件夹

# 2. 如果pic文件夹不存在，创建它
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

# 3. 保存为PDF文件
# 保存路径
pdf_path_normal = os.path.join(pic_dir, 'davar_normal.pdf')
pdf_path_abnormal = os.path.join(pic_dir, 'davar_abnormal.pdf')

# 保存 PDF
fig1.savefig(pdf_path_normal, format='pdf', bbox_inches='tight')
fig2.savefig(pdf_path_abnormal, format='pdf', bbox_inches='tight')

# 4. 显示图片
plt.show()


