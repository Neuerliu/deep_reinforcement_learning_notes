import matplotlib.pyplot as plt

# 定义正方形的四个顶点
points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]

# 分别获取x和y的坐标
x = [point[0] for point in points]
y = [point[1] for point in points]

# 绘制正方形
plt.plot(x, y, 'b-')  # 'b-'表示蓝色实线


plt.plot(x[0], y[0], 'go')  # 'go'表示绿色圆点
plt.plot(x[-2], y[-2], 'ro')  # 'ro'表示红色圆点

# 显示图形
plt.show()
