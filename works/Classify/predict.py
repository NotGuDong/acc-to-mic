# from models.MyResNet.ResNet import CreateResNet50
# from DL_FoodClassify.CreatDataset import creatDataset
# import torch
# # 单GPU或者CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# # 标签
# labels = {0:"apple_pie", 1:"baby_back_ribs", 2:"baklava", 3:"beef_carpaccio", 4:"beef_tartare"}
# # 种类
# classes = 5
#
# models = CreateResNet50(classes).to(device)
# models.load_state_dict(torch.load('../weights/best.pth', map_location=torch.device('cpu')))
# models.eval()
#
#
# test_dir = '../dataset/test'
# test_dataset = creatDataset(test_dir)
#
# cor = 0
# toa = 0
# for data, label in test_dataset:
#     data = data.unsqueeze(0)
#     output = models(data)
#     _, predicted = torch.max(output.data, 1)
#     toa += 1
#     if predicted.item() == label:
#         cor += 1
#     print("预测结果：{},真实结果：{}".format(labels[predicted.item()],  labels[label]))
# print("acc",cor/toa)