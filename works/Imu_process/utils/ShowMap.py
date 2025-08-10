from matplotlib import pyplot as plt

# 画出三轴时域图像，在形参中可改变标题名
def showMap(t_list, x_list, y_list, z_list, title='Original Map'):
    """
    :param t_list: 时间列表
    :param x_list: x轴加速度列表
    :param y_list: y轴加速度列表
    :param z_list: z轴加速度列表
    :param title: 标题
    :param ylim: 设置y轴的范围
    :return: None
    """
    # 用红、绿、蓝分别代表x轴、y轴和z轴数据
    colors = ['red', 'green', 'blue']
    lists = [x_list, y_list, z_list]
    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t_list, lists[i], color=colors[i])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    plt.show()