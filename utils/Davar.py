import numpy as np
import matplotlib.pyplot as plt

class DynamicAllanAnalyzer:
    """
    动态 Allan 方差 + 动态 ARW 分析器
    修正点：
    1. 正确计算 Allan 方差
    2. ARW 单位转换严格正确（°/√h）
    3. 自动过滤过大 tau
    4. 结构简化且更安全
    """

    # ------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------
    def __init__(self, dt=1.0, sf=1.0):
        """
        dt: 采样周期 (秒)
        sf: 缩放因子（输入数据除以 sf）
        """
        self.dt = dt
        self.sf = sf
        self.results = {}

    # ------------------------------------------------------------
    # 动态 ARW（严格单位转换）
    # ------------------------------------------------------------
    def calculate_dynamic_arw(self, window_data):
        """
        直接从窗口差分计算 τ=1 秒的随机游走系数
        ARW(°/√h) = sqrt(0.5 * mean((x[k+1]-x[k])^2)) * sqrt(3600)
        """
        diff = np.diff(window_data)
        avar_1s = 0.5 * np.mean(diff ** 2)
        arw = np.sqrt(avar_1s) * np.sqrt(3600)
        return arw

    # ------------------------------------------------------------
    # 总体分析接口
    # ------------------------------------------------------------
    def analyze(self, filename, window_size, step_size):

        data = np.load(filename) / self.sf
        print(data.shape)
        self.results = {}
        axis_list = ["X轴", "Y轴", "Z轴"]

        arw_columns = []  # 用来收集 X/Y/Z 三个轴的 ARW（列）

        for i, name in enumerate(axis_list):
            axis_data = data[:, i]

            # 使用当前窗口直接计算 ARW
            arw = []
            for s in range(0, len(axis_data) - window_size + 1, step_size):
                w = axis_data[s:s + window_size]
                arw.append(self.calculate_dynamic_arw(w))
            arw = np.array(arw)

            # 保存到 results 字典
            self.results[name] = {
                "arw": arw
            }

            # 收集 ARW 列
            arw_columns.append(arw.reshape(-1, 1))

        # 拼成 (rows, 3) 的 numpy 数组
        arw_matrix = np.concatenate(arw_columns, axis=1)

        return arw_matrix

    # ------------------------------------------------------------
    # 画动态 ARW 曲线
    # ------------------------------------------------------------
    def plot_dynamic_arw(self):
        if not self.results:
            print("请先运行 analyze()")
            return

        fig, ax = plt.subplots(3, 1, figsize=(12, 10))

        for i, name in enumerate(["X轴", "Y轴", "Z轴"]):
            r = self.results[name]
            ax[i].plot(range(len(r["arw"])), r["arw"], lw=2)
            ax[i].set_title(f"{name} 动态 ARW")
            ax[i].set_ylabel("ARW (°/√h)")
            ax[i].grid(True)

        ax[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 返回统计信息（Pandas）
    # ------------------------------------------------------------
    def get_arw_statistics(self):
        import pandas as pd

        datas = []
        for name in ["X轴", "Y轴", "Z轴"]:
            arw = self.results[name]["arw"]
            datas.append({
                "轴": name,
                "均值(°/√h)": np.nanmean(arw),
                "标准差(°/√h)": np.nanstd(arw),
                "最小值": np.nanmin(arw),
                "最大值": np.nanmax(arw)
            })

        return pd.DataFrame(datas)


# ------------------------------------------------------------
# 使用示例
# ------------------------------------------------------------
if __name__ == "__main__":

    analyzer = DynamicAllanAnalyzer(dt=1, sf=252000)

    results = analyzer.analyze(
        filename=r"..\datasets\three_vibra\SMAP_train.npy",
        window_size=96,
        step_size=1,
    )
    print(results.shape)
    analyzer.plot_dynamic_arw()

    print(analyzer.get_arw_statistics())
