import numpy as np
import pandas as pd


def fileread_csv(FileName, sr):
    df = pd.read_csv(FileName + '.csv', header=None)
    # 将tsv_reader中采样点数量储存在data_line_number中
    data_line_number = int(len(df.iloc[:, 0].tolist()))
    # 下方500代表手机内置加速度传感器的采样率，根据不同加速度传感器的采样率可灵活修改
    timelong = data_line_number / sr
    # 求出时间序列t_list，并读取三轴加速度信号x_list、y_list、z_list
    t_list = list(np.linspace(0, timelong, int(data_line_number)))

    x_list = df.iloc[:, 1].tolist()
    y_list = df.iloc[:, 2].tolist()
    z_list = df.iloc[:, 3].tolist()

    return t_list, x_list, y_list, z_list

def wavread_csv(FileName, sr):
    df = pd.read_csv(FileName + '.csv', header=None)
    # 将tsv_reader中采样点数量储存在data_line_number中
    data_line_number = int(len(df.iloc[:, 0].tolist()))
    # 下方500代表手机内置加速度传感器的采样率，根据不同加速度传感器的采样率可灵活修改
    timelong = data_line_number / sr
    # 求出时间序列t_list，并读取三轴加速度信号x_list、y_list、z_list
    t_list = list(np.linspace(0, timelong, int(data_line_number)))

    x_list = df.iloc[:, 1].tolist()
    y_list = df.iloc[:, 2].tolist()
    z_list = df.iloc[:, 3].tolist()

    return t_list, x_list, y_list, z_list