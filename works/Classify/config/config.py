
dataset = "phoneAB_C"
weights = "phoneAB_C"
# resnet50\
model = 'densenet121'

configs = {
    "model_config":
        {"model_name": model},

    "train_config":
        {"batch_size": 8,
         "epochs": 100,
         "learining_rate": 0.001,
         "momentum": 0.8,
         "optimizer": 'Adam',
         "loss": 'CrossEntropyLoss',
         "subsequent_training":  False,
         "classes": 7
         },


    "data_config":
        {"checkpoint_path": './checkpoint',
         "checkpoint": 'checkpoint_3.pth',
         "train_dir": 'dataset/' + dataset +'/train',
         "val_dir": 'dataset/' + dataset +'/val',
         "log_dir": './log/' + dataset +'_' + model,
         "best_model_weights": './weights/' + dataset +'_' + model +'_lr0001_momentum9.pth',
         "last_model_weights": './weights/' + dataset +'_' + model +'_last.pth',
         "weights": './weights/' + weights + '_densenet121_lr0001_momentum9.pth'}
}