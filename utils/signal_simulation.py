from arw_simulation import  signal1, signal2, step_position
import matplotlib.pyplot as plt
import numpy as np


# 创建图形
plt.figure(figsize=(12, 8))

# 绘制 signal1
plt.subplot(2, 1, 1)
plt.plot(signal1)
plt.title('Signal 1: White Noise')
plt.ylabel('Amplitude')
plt.grid(True)

# 绘制 signal2
plt.subplot(2, 1, 2)
plt.plot(signal2)
plt.title('Signal 2: White Noise + Step Change in Variance')
plt.xlabel('Time (hours)')
plt.ylabel('Amplitude')
plt.grid(True)

# 在signal2图中标记台阶位置
step_time = step_position / 3600
plt.axvline(x=step_time, color='red', linestyle='--', alpha=0.7, label=f'Step at {step_time:.2f}h')
plt.legend()

plt.tight_layout()
plt.show()