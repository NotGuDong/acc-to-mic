import numpy as np

# Z-score归一化
# 该方法使用原始数据的均值和标准差进行标准化，处理后的数据符合标准正态分布，即均值为0，标准差为1
def standardize(t_list, x_list, y_list, z_list):
    lists = [x_list, y_list, z_list]
    for i in range(0, 3):
        lists[i][:] = np.subtract(lists[i], np.mean(lists[i])) / np.var(lists[i])
    return t_list, x_list, y_list, z_list