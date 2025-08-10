import pandas as pd
from utils import FileRead
from utils import ShowMap
import utils.preprocess as pre
import utils.SegmentationPro as seg
import utils.ShowCuttingPro as ShowCuttingPro
import utils.Spectrogram as sp
import numpy as np
from scipy.stats import kurtosis
import csv


if __name__ == "__main__":
    # prefix = r"/Users/lance/Downloads/python/Dataset/IMU_dataset/"
    prefix = "D:/Data/"
    num = 1
    name = "objection"  # (" + str(num) + ")"
    classes = ["password", "Mike", "goodbye", "hello", "address", "answer", "tomato", "potato", "China", "America"]
    theClass = classes[num-1]

    sr = 417  # 采样率

    # 读取原始数据，将时间、x轴、y轴、z轴分别储存在tList, xList, yList, zList中
    tList, xList, yList, zList = FileRead.fileread_csv(prefix + name, sr)

    # 通过showMap函数绘制原始三轴加速度信号图
    # ShowMap.showMap(tList, xList, yList, zList)

    # 拷贝原始数据
    t, x, y, z = (tList.copy(), xList.copy(), yList.copy(), zList.copy())

    # 信号两端波动数据的去除、归一化、插值和高通滤波
    # Normalized=1表明进行归一化处理
    # highpass=30/500表示进行截止频率为30Hz的高通滤波
    tList, fList, xList, yList, zList = pre.preprocess(tList, xList, yList, zList, Normalized=1, highpass=30 / sr)

    # 通过showMap函数绘制进行预处理后的三轴加速度信号图
    ShowMap.showMap(tList, xList, yList, zList, '30Hz filter')
    print("read successfully")
    # 做平滑处理（仅对z轴进行）
    zSmooth, tSmooth = seg.smooth(zList, tList)
    for i in range(0, 4):
        zSmooth, tSmooth = seg.smooth(zSmooth, tSmooth)

    # 找寻切割点
    # 设置切割阈值为0.8 * maxValue + 0.2 * minValue
    cuttingpoints = seg.findCuttingPoints(tSmooth, zSmooth, (0.3, 0.55))
    # cuttingpoints[1] += 200
    # cuttingpoints[13] += 300
    # del cuttingpoints[12]
    # del cuttingpoints[13]
    # cuttingpoints[18] += 450
    # cuttingpoints.insert(0, -200)
    # cuttingpoints.insert(19, 27000)
    # cuttingpoints.insert(19, 7200)

    # 原始拷贝的信号也需要做信号两端波动数据的去除、归一化、插值和高通滤波

    t, f, x, y, z = pre.preprocess(t, x, y, z, Normalized=1, highpass=30 / sr)
    # 在原始拷贝的信号上显示切割点，并将所有切割点保存在"Picture"文件夹内
    # left和right表示令两个切割点分别向左和向右移动left和right个单位以完全覆盖被切割信号
    left = 0
    right = 0

    try:
        for i in range(0, 10):
            i = 2*i
            cuttingpoints[i] = cuttingpoints[i] + 350
            cuttingpoints[i + 1] = cuttingpoints[i+1] + 750
    except:
        pass

    path = prefix + "Picture/" + name + ".png"
    ShowCuttingPro.showcutting(cuttingpoints, t, x, y, z, left, right, path)

    # 输出cuttingpoints的值，以便在需要的时候人工调整切割点
    print("cuttingpoints的值为:" + str(cuttingpoints))


    # 画时频图、RGB图，保存切割后的csv文件
    try:
        for i in range(0, 10):
            i = 2 * i
            # 切割x、y和z轴信号时需要判断是否越出边界
            x_new = x[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            y_new = y[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            z_new = z[cuttingpoints[i] - left if cuttingpoints[i] - left >= 0 else 0: cuttingpoints[i + 1] + right if
            cuttingpoints[i + 1] + right <= len(t) - 1 else len(t) - 1]
            dic = {"z": z_new}
            df = pd.DataFrame(data=dic)
            # 将切割后的z轴信号储存在同路径下，名称例如"ty_s9_account_align_0.csv"
            # df.to_csv(prefix + name + "_align_" + str(int(i / 2)) + ".csv", index=False, header=False, sep="\t")

            # 绘制ASD图
            # sp.generateASD(x_new, y_new, z_new, name)

            # 绘制出z轴频谱图，储存在"/Spectrogram/"路径下，名称例如"ty_s9_account_Spectrogram_0.pdf/.png"
            x_new, y_new, z_new = sp.generateMap(prefix + "Spectrogram/" + name + "_Spectrogram_" + str(int(i / 2)),
                                                x_new, y_new, z_new, name)
            # 绘制出三轴RGB图，储存在"/RGB/"路径下，名称例如"ty_s9_account_0.png"
            # if i/2 < 9:
            #     sp.generateRGB(x_new, y_new, z_new, prefix + "RGB/phoneA/train/" + theClass + "/" + name + "_" + str(int(i / 2)) + ".png")
            # else:
            sp.generateRGB(x_new, y_new, z_new, prefix + "RGB/phoneF/train/" + theClass + "/" + name + "_" + str(int(i / 2)) + ".png")

    except:
        pass
