import numpy as np
from scipy import signal
import utils.denoise as denoise

# TODO: 这里是高通滤波器，通过对信号进行短时傅里叶变换，相关区间数据置零并进行短时傅里叶逆变换来实现高通滤波
def highPassFilter(xList, yList, zList, thresRate):
    # 由于之前将加速度传感器的采样率统一至1000Hz，这里fs设置为采样率的一半，即1000 / 2 = 500
    fs = 500
    # 下方nperseg代表窗函数的长度，noverlap代表窗函数重叠数，return_onesided=True表示返回复数实部
    freqsX, tx, resultX, = signal.stft(xList, fs, nperseg=128, noverlap=120, window="hann", boundary=None, padded=False,
                                       return_onesided=True)
    freqsY, ty, resultY = signal.stft(yList, fs, nperseg=128, noverlap=120, window="hann", boundary=None, padded=False,
                                      return_onesided=True)
    freqsZ, tz, resultZ = signal.stft(zList, fs, nperseg=128, noverlap=120, window="hann", boundary=None, padded=False,
                                      return_onesided=True)
    thres = np.ceil(resultZ.shape[0] * thresRate)

    # 这里进行的是滤波操作，将阈值以下的频率分量置零，滤波的具体参数由thresRate控制
    for i in range(0, resultZ.shape[0]):
        if i <= thres or i >= resultZ.shape[0] - thres:
            resultX[i, :] = 0
            resultY[i, :] = 0
            resultZ[i, :] = 0

    # 进行傅里叶逆变换，求出高通滤波后的信号
    t, resultX = signal.istft(resultX, fs, nfft=128, noverlap=120)
    t, resultY = signal.istft(resultY, fs, nfft=128, noverlap=120)
    t, resultZ = signal.istft(resultZ, fs, nfft=128, noverlap=120)

    resultX = denoise.wavelet_noising(resultX)
    resultY = denoise.wavelet_noising(resultY)
    resultZ = denoise.wavelet_noising(resultZ)

    return t, freqsZ, resultX, resultY, resultZ