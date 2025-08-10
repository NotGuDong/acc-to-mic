import numpy as np
import matplotlib.pyplot as plt


# 平滑处理
def smooth(zList, tList):
    zList_tmp = zList.copy()
    length = len(zList)
    # 这里设置两轮平滑处理，滑动窗口长度分别设置为200和30
    config1 = 200
    config2 = 30
    newSize = config1 + config2 - 2
    tSmooth = tList[(int)(newSize / 2):length - (int)(newSize / 2)]
    zSmooth = [0 for i in range(0, length - newSize)]
    zList_tmp[:] = np.abs(zList_tmp)
    zSmooth[:] = np.convolve(zList_tmp, np.ones(config1) / config1, mode='valid')
    zSmooth[:] = np.convolve(zSmooth, np.ones(config2) / config2, mode='valid')
    return zSmooth, tSmooth


# 平滑后找寻切割点
def findCuttingPoints(tSmooth, zSmooth, para):
    zSmooth = np.array(zSmooth)
    # 求出进行平滑处理后z轴信号的最高值Mmax和最低值Mmin
    Mmax = np.max(zSmooth)
    Mmin = np.min(zSmooth)
    leng = zSmooth.size
    # 设定切割的阈值，例如设置阈值为0.8*Mmax + 0.2*Mmin
    thres = para[1] * Mmin + para[0] * Mmax
    # cuttingpoints将存储所有切割点的坐标，两两一对
    cuttingpoints = []
    # 分别画出z轴的平滑信号和阈值线
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 3), sharey=True)
    ax.set_title('Smoothed signal along the z-axis', fontsize=30)
    plt.ylabel('Acc (m/s\N{SUPERSCRIPT TWO})', fontsize=30, labelpad=34)
    plt.xlabel('Time (secs)', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    y = []
    for i in range(0, len(tSmooth)):
        y.append(thres)
    ax.plot(tSmooth, zSmooth, lw=5)
    ax.plot(tSmooth, y, lw=5)

    # 求出所有z轴平滑信号和阈值线相交样本点的坐标，并存储在cuttingpoints
    # cuttingpoints的一个样例为：[2845, 3777, 6591, 7727, 10859, 12051, 15206, 16302, 19510, 20584, 23802, 24851, 28185, 29193, 32542, 33514, 36814, 37991, 41313, 42263]，两两一对
    for i in range(0, leng - 1):
        if zSmooth[i] <= thres and zSmooth[i + 1] > thres:
            cuttingpoints.append(i)
        elif zSmooth[i] >= thres and zSmooth[i + 1] < thres:
            cuttingpoints.append(i)

    # 使用cuttingpoints数据在上图中画出分割点
    for i in cuttingpoints:
        plt.scatter(tSmooth[i], zSmooth[i], color='red', linewidths=10)
    plt.figure(figsize=(16, 3))
    fig.show()
    return cuttingpoints