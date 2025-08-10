import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal
import cv2
from scipy.fft import rfft, rfftfreq


# 求出三个轴的短时傅里叶变换并画出相应的频谱图
def generateMap(filename, xList, yList, zList, name="Magnitude Scalogram"):
    try:
        # 汉明窗的窗长
        NFFT = 128
        # 每次窗口移动的距离为128 - 120 = 8
        noverlap = 126

        # 采样率为1000
        Fs = 1000
        fig = plt.figure(figsize=(9, 4))
        plot_Z = fig.subplots(ncols=1)

        # 进行短时傅里叶变换
        fX, tX, specX = signal.stft(xList, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, detrend=False)
        fY, tY, specY = signal.stft(yList, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, detrend=False)
        fZ, tZ, specZ = signal.stft(zList, fs=Fs, window='hann', nperseg=NFFT, noverlap=noverlap, detrend=False)

        # 将相应的短时傅里叶变换结果取绝对值
        specX = np.abs(specX)
        specY = np.abs(specY)
        specZ = np.abs(specZ)

        # 画出二维色彩图
        plot_Z.pcolormesh(tZ, fZ, specZ)
        plot_Z.set_xlabel("Time (secs)", fontsize=30)
        plot_Z.set_ylabel("Frequency (Hz)", fontsize=30)
        plt.xticks(fontsize=20)
        plt.title(name)
        plt.yticks(fontsize=20)
        plt.savefig(filename + ".png")
        # plt.savefig(filename + ".pdf", bbox_inches='tight')
        plt.show()
        return specX, specY, specZ
    except:
        pass


# 将z轴的频谱图转换为相应的RGB图像
def generateRGB(specX, specY, specZ, name):
    R_x = np.zeros(shape=specX.shape)
    G_y = np.zeros(shape=specX.shape)
    B_z = np.zeros(shape=specX.shape)

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            R_x[i, j] = math.sqrt(specX[i, j])
    maxi = np.max(R_x)
    mini = np.min(R_x)
    # 将矩阵中每个点映射到0-255之间
    R_x = np.ceil((255 / (maxi - mini)) * (R_x - mini))

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            G_y[i, j] = math.sqrt(specY[i, j])
    maxi = np.max(G_y)
    mini = np.min(G_y)
    G_y = np.ceil((255 / (maxi - mini)) * (G_y - mini))

    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            B_z[i, j] = math.fabs(specZ[i, j])
    maxi = np.max(B_z)
    mini = np.min(B_z)
    B_z = np.ceil((255 / (maxi - mini)) * (B_z - mini))

    img = np.zeros(shape=(specX.shape[0], specX.shape[1], 3))
    for i in range(specX.shape[0]):
        for j in range(specX.shape[1]):
            # 三轴信息
            img[i, j, 2] = R_x[i, j]
            img[i, j, 1] = G_y[i, j]
            img[i, j, 0] = B_z[i, j]
    # plt.savefig(name)
    cv2.imwrite(name, img)

    # img_R = np.zeros(shape=(specX.shape[0], specX.shape[1], 3))
    # for i in range(specX.shape[0]):
    #     for j in range(specX.shape[1]):
    #         # RGB图像仅取x轴的信息，以降低其他两个轴对实验结果的干扰
    #         img_R[i, j, 2] = R_x[i, j]
    #         img_R[i, j, 1] = 0
    #         img_R[i, j, 0] = 0
    # # plt.savefig(name)
    # cv2.imwrite(name + 'R.png', img_R)
    #
    # img_G = np.zeros(shape=(specX.shape[0], specX.shape[1], 3))
    # for i in range(specX.shape[0]):
    #     for j in range(specX.shape[1]):
    #         # RGB图像仅取y轴的信息，以降低其他两个轴对实验结果的干扰
    #         img_G[i, j, 2] = 0
    #         img_G[i, j, 1] = G_y[i, j]
    #         img_G[i, j, 0] = 0
    # # plt.savefig(name)
    # cv2.imwrite(name + 'G.png', img_G)
    #
    # img_B = np.zeros(shape=(specX.shape[0], specX.shape[1], 3))
    # for i in range(specX.shape[0]):
    #     for j in range(specX.shape[1]):
    #         # RGB图像仅取z轴的信息，以降低其他两个轴对实验结果的干扰
    #         img_B[i, j, 2] = 0
    #         img_B[i, j, 1] = 0
    #         img_B[i, j, 0] = B_z[i, j]
    # # plt.savefig(name)
    # cv2.imwrite(name, img_B)

def generateASD(X_list, Y_list, Z_list, name):

    asd_x = rfft(X_list)
    asd_y = rfft(Y_list)
    asd_z = rfft(Z_list)

    fre_x = rfftfreq(len(X_list),1 / 1000)
    fre_y = rfftfreq(len(Y_list), 1 / 1000)
    fre_z = rfftfreq(len(Z_list), 1 / 1000)

    plt.plot(fre_z, np.abs(asd_z))
    plt.title(name + "  ASD")
    plt.show()



