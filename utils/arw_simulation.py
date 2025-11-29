import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'   # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False        # 允许负号显示

class DynamicAllanAnalyzer:
    """
    动态 Allan 方差 + 动态 ARW 分析器（窗口差分法）
    """
    def __init__(self):
        self.results = {}

    def calculate_dynamic_arw(self, window_data):
        # 直接从窗口差分计算 τ=1 秒的随机游走系数
        diff = np.diff(window_data)
        avar_1s = 0.5 * np.mean(diff ** 2)
        arw = np.sqrt(avar_1s) * 60
        return arw

    def analyze_signal(self, signal, window_size=50, step_size=1):
        arw = []
        for s in range(0, len(signal) - window_size + 1, step_size):
            w = signal[s:s + window_size]
            arw.append(self.calculate_dynamic_arw(w))
        return np.array(arw)

# ------------------------------------------------------------
# 生成信号
# ------------------------------------------------------------
N = 60000
sigma1 = 2   # 前半段噪声方差
sigma2 = 3 # 后半段噪声方差
step_position = 40000

# 信号1：纯白噪声
signal1 = np.random.normal(0, sigma1, N)  / 3600

# 信号2：白噪声 + 台阶 + 不同方差
noise_before = np.random.normal(0, sigma1, step_position)
noise_after = np.random.normal(0, sigma2, N - step_position)
signal2 = np.concatenate([noise_before, noise_after]) / 3600

### 和标度因数没关系，都是换成°/s，标准单位进去计算

if __name__ == "__main__":

    # ------------------------------------------------------------
    # 动态 ARW 分析
    # ------------------------------------------------------------
    analyzer = DynamicAllanAnalyzer()

    window_size = 100
    step_size = 1

    arw1 = analyzer.analyze_signal(signal1, window_size, step_size)
    arw2 = analyzer.analyze_signal(signal2, window_size, step_size)
    # ------------------------------------------------------------
    # pic2: 动态 ARW
    # ------------------------------------------------------------
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(np.arange(len(arw1)) / 100, arw1)
    plt.xlabel("Time")
    plt.ylabel("N")
    plt.grid(False)

    plt.subplot(1,2,2)
    plt.plot(np.arange(len(arw2))/100, arw2)
    plt.xlabel("Time")
    plt.ylabel("N")
    plt.grid(False)

    plt.tight_layout()
    plt.show()






