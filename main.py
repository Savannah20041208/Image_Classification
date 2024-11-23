import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib

# 设置中文字体以支持中文标题和标签
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_curves(t_loss, v_loss, t_acc, v_acc):
    """绘制训练和验证的损失和准确率曲线"""
    epochs = range(1, len(t_loss) + 1)
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, t_loss, label="训练损失")
    plt.plot(epochs, v_loss, label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失值")
    plt.title("训练和验证损失曲线")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, t_acc, label="训练准确率")
    plt.plot(epochs, v_acc, label="验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.title("训练和验证准确率曲线")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.show()


def train_model(model, t_loader, v_loader, criterion, optimizer, scheduler, epochs=50, patience=15):
    best_acc = 0.0   # 初始化最佳验证准确率
    no_improve_epochs = 0   # 用于记录验证准确率未提升的连续轮数
    t_loss_history, v_loss_history = [], []
    t_acc_history, v_acc_history = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()  # 设置为训练模式
        running_loss, running_corrects = 0.0, 0

        # 遍历训练数据
        for inputs, lab in t_loader:
            inputs, lab = inputs.to(device), lab.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, lab)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == lab.data)

        # 计算训练损失和准确率
        t_loss = running_loss / len(t_loader.dataset)
        t_acc = running_corrects.double() / len(t_loader.dataset)

        # 验证阶段
        model.eval()
        v_loss, v_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, lab in v_loader:
                inputs, lab = inputs.to(device), lab.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, lab)
                _, preds = torch.max(outputs, 1)
                v_loss += loss.item() * inputs.size(0)  # 累计验证损失
                v_corrects += torch.sum(preds == lab.data)  # 累计正确预测样本数


        v_loss = v_loss / len(v_loader.dataset)
        val_acc = v_corrects.double() / len(v_loader.dataset)

        print(f"训练损失: {t_loss:.4f}, 训练准确率: {t_acc:.4f}, 验证损失: {v_loss:.4f}, 验证准确率: {val_acc:.4f}")

        scheduler.step(v_loss)

        # 更新最佳模型
        if val_acc > best_acc:  #如果性能提高
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:  # 检查早停条件
            print(f"早停: 验证准确率在 {patience} 个 epoch 内没有提升。")
            break

        # 记录历史损失和准确率
        t_loss_history.append(t_loss)
        v_loss_history.append(v_loss)
        t_acc_history.append(t_acc.item())
        v_acc_history.append(val_acc.item())

    print(f"最佳验证准确率: {best_acc:.4f}")
    return t_loss_history, v_loss_history, t_acc_history, v_acc_history


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理和加载
    data_dir = r"D:\Pytorch\dataset"
    batch_size = 32
    img_size = 224

    weights = ResNet50_Weights.DEFAULT  # 加载预训练权重

    transform = {  # 定义数据增强
        'train': transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomVerticalFlip(),

            transforms.RandomErasing(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        ])
    }

    # 加载数据集并划分
    full_dataset = datasets.ImageFolder(data_dir)
    t_size = int(0.8 * len(full_dataset))  # 80% 训练集
    v_size = len(full_dataset) - t_size   # 20% 验证集
    t_dataset, v_dataset = torch.utils.data.random_split(
        full_dataset, [t_size, v_size], generator=torch.Generator().manual_seed(42)
    )

    # 应用数据增强
    t_dataset.dataset.transform = transform['train']
    v_dataset.dataset.transform = transform['val']

    # 加载数据
    t_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    v_loader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = full_dataset.classes
    print(f"分类类别: {class_names}")

    # 构建模型
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model = model.to(device)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 开始训练
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
        model, t_loader, v_loader, criterion, optimizer, scheduler, epochs=100, patience=15
    )

    # 绘制训练曲线
    plot_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

    # 模型评估
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in v_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 分类报告
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 绘制混淆矩阵
    plot_matrix(y_true, y_pred, class_names)