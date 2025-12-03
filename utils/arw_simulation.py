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
sigma1 = 2   # 第一段和第三段噪声方差
sigma2 = 3   # 第二段噪声方差
step_position1 = 20000  # 第一段结束位置
step_position2 = 50000  # 第二段结束位置（第三段开始位置）

# 信号1：纯白噪声（如果需要保持原样）
signal1 = np.random.normal(0, sigma1, N) / 3600

# 信号2：三段不同方差的白噪声
# 第一段：方差为 sigma1
noise_segment1 = np.random.normal(0, sigma1, step_position1)

# 第二段：方差为 sigma2
segment2_length = step_position2 - step_position1
noise_segment2 = np.random.normal(0, sigma2, segment2_length)

# 第三段：方差为 sigma1（与第一段相同）
segment3_length = N - step_position2
noise_segment3 = np.random.normal(0, sigma1, segment3_length)

# 合并三段信号
signal2 = np.concatenate([noise_segment1, noise_segment2, noise_segment3]) / 3600

### 和标度因数没关系，都是换成°/s，标准单位进去计算

# ------------------------------------------------------------
# 动态 ARW 分析
# ------------------------------------------------------------
analyzer = DynamicAllanAnalyzer()

window_size = 100
step_size = 1

arw1 = analyzer.analyze_signal(signal1, window_size, step_size)
arw2 = analyzer.analyze_signal(signal2, window_size, step_size)

if __name__ == "__main__":
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






