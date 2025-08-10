import utils.DataRemove as DataRemove
import utils.Standardize as Standardize
import utils.Interp as Interp
import utils.HighPassFilter as HighPassFilter

# 预处理信号，分别为：信号两端波动数据的去除、归一化、插值和高通滤波
def preprocess(tList, xList, yList, zList, Normalized=0, highpass=-1):
    # 信号两端波动数据的去除
    tList, xList, yList, zList = DataRemove.dataRemove(tList, xList, yList, zList, start=0, end=-10)

    # 归一化
    if Normalized == 1:
        tList, xList, yList, zList = Standardize.standardize(tList, xList, yList, zList)

    # 插值
    tList, xList, yList, zList = Interp.interp(tList, xList, yList, zList)

    # 高通滤波
    if highpass >= 0:
        tList, fList, xList, yList, zList = HighPassFilter.highPassFilter(xList, yList, zList, highpass)
        return tList, fList, xList, yList, zList
    return tList, xList, yList, zList