from matplotlib import pyplot as plt

# 绘制原始信号被预处理后的信号及相应的切割点，并将时域图保存在path路径下
def showcutting(cuttingpoints, t_x_list_filter_two, x_list_filter_two, y_list_filter_two, z_list_filter_two, left,
                right, path):
    cuttingpoints_two = cuttingpoints[:]
    try:
        for i in range(len(cuttingpoints_two)):
            if (i % 2 == 0):
                # 下述代码用于判断边界条件
                if (cuttingpoints_two[i] - left <= 0):
                    cuttingpoints_two[i] = 0
                else:
                    cuttingpoints_two[i] = cuttingpoints_two[i] - left
                if (cuttingpoints_two[i + 1] + right > len(t_x_list_filter_two) - 1):
                    cuttingpoints_two[i + 1] = len(t_x_list_filter_two) - 1
                else:
                    cuttingpoints_two[i + 1] = cuttingpoints_two[i + 1] + right
    except:
        pass
    plt.plot(t_x_list_filter_two[:], z_list_filter_two[:], t_x_list_filter_two[cuttingpoints_two],
             z_list_filter_two[cuttingpoints_two], '*', markersize=20, lw=1)
    plt.title('Filtered signal with cutting points (z-axis)', fontsize=30)
    plt.xlabel('Time (secs)', fontsize=30)
    plt.ylabel('Acc (m/s\N{SUPERSCRIPT TWO})', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(path, bbox_inches='tight')
    plt.show()
