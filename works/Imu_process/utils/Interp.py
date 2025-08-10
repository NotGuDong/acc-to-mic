import numpy as np
from scipy import interpolate

# 插值
def interp(t_list, x_list, y_list, z_list):
    # 下方的1000代表将原有加速度传感器的采样率统一插值到1000Hz
    t_list_new = np.linspace(0, t_list[-1], int(t_list[-1] * 1000))
    fx = interpolate.interp1d(t_list, x_list, kind='slinear')
    x_list_new = fx(t_list_new)
    fy = interpolate.interp1d(t_list, y_list, kind='slinear')
    y_list_new = fy(t_list_new)
    fz = interpolate.interp1d(t_list, z_list, kind='slinear')
    z_list_new = fz(t_list_new)
    return t_list_new, x_list_new, y_list_new, z_list_new