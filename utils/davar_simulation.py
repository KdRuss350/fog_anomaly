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

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_davar_3d(t1/100, tau1/100, davar1, ax=ax1)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_davar_3d(t2/100, tau2/100, davar2, ax=ax2)


plt.tight_layout()
plt.show()


