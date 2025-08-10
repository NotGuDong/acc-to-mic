# 信号两端波动数据的去除
# start和end分别控制两端波动数据去除的样本点数量
def dataRemove(tList, xList, yList, zList, start=0, end=0):
    tList = tList[0:  end - start]
    xList = xList[start: end]
    yList = yList[start: end]
    zList = zList[start: end]
    return tList, xList, yList, zList