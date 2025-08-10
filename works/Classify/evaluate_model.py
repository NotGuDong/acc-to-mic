import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, dataset):
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    loader = DataLoader(dataset, batch_size=1)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = out.argmax(1).item()
            y_pred.append(pred)
            y_true.append(y.item())

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Keyword", "Keyword"]))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Keyword", "Keyword"], yticklabels=["Non-Keyword", "Keyword"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    dataset = AccDataset(CSV_DIR, LABEL_FILE, KEYWORDS, SEQ_LEN)
    _, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    evaluate_model("acc_hotword_model.pth", val_set)