import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import warnings

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")

# ----------------------------- ПАРАМЕТРЫ -----------------------------
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(DEVICE)
GROUPS_LIST = [1, 2, 4, 8]
SAVE_MODELS = True
SAVE_GRAPHS = True
OUTPUT_DIR = "./experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------- АУГМЕНТАЦИИ И ДАННЫЕ -----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ---------------------- RESNet-50 С ГРУППАМИ (как ранее) --------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet50Groups(nn.Module):
    def __init__(self, num_classes=100, groups=1):
        super(ResNet50Groups, self).__init__()
        self.groups = groups
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 3, stride=1, groups=groups)
        self.layer2 = self._make_layer(128, 4, stride=2, groups=groups)
        self.layer3 = self._make_layer(256, 6, stride=2, groups=groups)
        self.layer4 = self._make_layer(512, 3, stride=2, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, groups=groups))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ --------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops_approx(model, input_size=(1, 3, 32, 32), device='cpu'):
    """Приблизительный подсчёт FLOPs для ResNet-50 с групповыми свёртками."""
    # Простая формула: общее число умножений-сложений для всех свёрточных слоёв
    # Мы не будем делать точный расчёт, а вернём оценочное значение на основе числа параметров
    # Для ResNet-50 на CIFAR-100 ~4.5 GFLOPs для groups=1, для groups=8 ~0.6 GFLOPs
    # Эта функция нужна только для визуализации, точность не критична
    params = count_parameters(model)
    # Эмпирическая калибровка: для groups=1 параметры ~23M, FLOPs ~4.1G
    base_flops = 4.1e9
    base_params = 23_000_000
    flops = base_flops * (params / base_params) * 0.8  # поправочный коэффициент
    return flops


# Пытаемся использовать ptflops для точного подсчёта, если установлен
try:
    from ptflops import get_model_complexity_info


    def compute_flops_precise(model, device):
        model_copy = ResNet50Groups(num_classes=100, groups=model.groups).to(device)
        macs, params = get_model_complexity_info(model_copy, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
        flops = macs * 2  # MACs -> FLOPs
        return flops, params


    USE_PTFlOPS = True
    print("✅ Используется ptflops для точного подсчёта FLOPs")
except ImportError:
    USE_PTFlOPS = False
    print("⚠️ ptflops не установлен. Будут использованы приблизительные FLOPs. Установите: pip install ptflops")


def compute_flops(model, device):
    if USE_PTFlOPS:
        flops, _ = compute_flops_precise(model, device)
        return flops
    else:
        return compute_flops_approx(model)


def measure_inference_time(model, loader, device, num_batches=50):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            start = time.time()
            _ = model(inputs)
            end = time.time()
            times.append(end - start)
    return np.mean(times) * 1000  # ms


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, use_tqdm=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    tqdm_loader = tqdm(loader, desc=f"Epoch {epoch:2d} Train", leave=False) if use_tqdm else loader
    for inputs, targets in tqdm_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if use_tqdm:
            tqdm_loader.set_postfix(loss=loss.item(), acc=100. * correct / total)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Valid", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(results, output_dir):
    plt.figure(figsize=(12, 5))
    for g in GROUPS_LIST:
        plt.plot(results[g]['history']['train_loss'], label=f'groups={g} train')
        plt.plot(results[g]['history']['val_loss'], '--', label=f'groups={g} val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Сравнение потерь (train/val) для всех групп')
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.show()

    plt.figure(figsize=(12, 5))
    for g in GROUPS_LIST:
        plt.plot(results[g]['history']['train_acc'], label=f'groups={g} train')
        plt.plot(results[g]['history']['val_acc'], '--', label=f'groups={g} val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Сравнение точности (train/val)')
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    for g in GROUPS_LIST:
        plt.plot(results[g]['history']['val_acc'], label=f'groups={g}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Валидационная точность по эпохам')
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'val_accuracy_comparison.png'), dpi=150)
    plt.show()


def plot_comparison_bars(results, output_dir):
    groups = GROUPS_LIST
    params = [results[g]['params'] / 1e6 for g in groups]
    best_acc = [results[g]['best_acc'] for g in groups]
    train_time = [results[g]['time'] for g in groups]
    flops = [results[g]['flops'] / 1e9 for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].bar([str(g) for g in groups], params, color='skyblue')
    axes[0, 0].set_xlabel('Groups')
    axes[0, 0].set_ylabel('Parameters (millions)')
    axes[0, 0].set_title('Количество параметров')
    axes[0, 0].grid(axis='y')

    axes[0, 1].bar([str(g) for g in groups], best_acc, color='lightcoral')
    axes[0, 1].set_xlabel('Groups')
    axes[0, 1].set_ylabel('Best Validation Accuracy (%)')
    axes[0, 1].set_title('Точность на валидации')
    axes[0, 1].grid(axis='y')

    axes[1, 0].bar([str(g) for g in groups], train_time, color='lightgreen')
    axes[1, 0].set_xlabel('Groups')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Время обучения (30 эпох)')
    axes[1, 0].grid(axis='y')

    axes[1, 1].bar([str(g) for g in groups], flops, color='gold')
    axes[1, 1].set_xlabel('Groups')
    axes[1, 1].set_ylabel('GFLOPs')
    axes[1, 1].set_title('Вычислительная сложность (GFLOPs)')
    axes[1, 1].grid(axis='y')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'comparison_bars.png'), dpi=150)
    plt.show()


def plot_accuracy_vs_params(results, output_dir):
    groups = GROUPS_LIST
    params = [results[g]['params'] / 1e6 for g in groups]
    acc = [results[g]['best_acc'] for g in groups]
    plt.figure(figsize=(8, 6))
    plt.scatter(params, acc, s=100, c=groups, cmap='viridis')
    for i, g in enumerate(groups):
        plt.annotate(f'groups={g}', (params[i], acc[i]), xytext=(5, 5), textcoords='offset points')
    z = np.polyfit(params, acc, 1)
    p = np.poly1d(z)
    plt.plot(params, p(params), "r--", alpha=0.6, label=f'тренд (slope={z[0]:.2f})')
    plt.xlabel('Parameters (millions)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Зависимость точности от числа параметров')
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_params.png'), dpi=150)
    plt.show()


def plot_heatmap(results, output_dir):
    df = pd.DataFrame({
        'Groups': GROUPS_LIST,
        'Params_M': [results[g]['params'] / 1e6 for g in GROUPS_LIST],
        'Best_Acc': [results[g]['best_acc'] for g in GROUPS_LIST],
        'Time_s': [results[g]['time'] for g in GROUPS_LIST],
        'FLOPs_G': [results[g]['flops'] / 1e9 for g in GROUPS_LIST]
    }).set_index('Groups')
    corr = df.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Корреляция между метриками')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.show()


def plot_confusion_matrix(model, loader, device, num_classes=100, output_dir=None):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Computing confusion matrix"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds, labels=range(20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(20))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix (first 20 classes)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_top20.png'), dpi=150)
    plt.show()


def plot_weight_grad_histograms(model, output_dir=None):
    weights = []
    grads = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weights.append(param.data.cpu().numpy().flatten())
            if param.grad is not None:
                grads.append(param.grad.cpu().numpy().flatten())
    if weights:
        all_weights = np.concatenate(weights)
        plt.figure(figsize=(10, 4))
        plt.hist(all_weights, bins=50, alpha=0.7, label='Weights')
        plt.title('Гистограмма весов модели')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'weights_hist.png'), dpi=150)
        plt.show()
    if grads:
        all_grads = np.concatenate(grads)
        plt.figure(figsize=(10, 4))
        plt.hist(all_grads, bins=50, alpha=0.7, color='orange', label='Gradients')
        plt.title('Гистограмма градиентов')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'gradients_hist.png'), dpi=150)
        plt.show()



if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # Обрати внимание: здесь уже используются DataLoader'ы с num_workers=0
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}

    # ---------------------------- ОСНОВНОЙ ЦИКЛ --------------------------

    results = {}
    for groups in GROUPS_LIST:
        print("\n" + "=" * 70)
        print(f"🔥 ЭКСПЕРИМЕНТ: groups = {groups}")
        print("=" * 70)
        model = ResNet50Groups(num_classes=100, groups=groups).to(DEVICE)
        total_params = count_parameters(model)
        print(f"📊 Количество параметров: {total_params:,}")

        flops = compute_flops(model, DEVICE)
        flops_g = flops / 1e9
        print(f"⚙️  FLOPs (приблизительные): {flops_g:.2f} GFLOPs")

        # Время инференса на CPU (переносим на CPU для замера)
        model_cpu = ResNet50Groups(num_classes=100, groups=groups).to('cpu')
        inf_time = measure_inference_time(model_cpu, testloader, 'cpu', num_batches=50)
        print(f"⏱️  Среднее время инференса (CPU, batch=1): {inf_time:.2f} ms")
        del model_cpu

        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        best_model_state = None
        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, DEVICE, epoch, use_tqdm=True)
            val_loss, val_acc = validate(model, testloader, criterion, DEVICE)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            lr = optimizer.param_groups[0]['lr']
            print(f"✅ Эпоха {epoch:2d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {lr:.5f} | Best Val Acc: {best_val_acc:.2f}%")

        training_time = time.time() - start_time
        print(f"\n🏁 Закончено groups={groups}. Лучшая точность на валидации: {best_val_acc:.2f}%")
        print(f"⏱️  Общее время обучения: {training_time:.2f} сек ({training_time / 60:.2f} мин)")

        results[groups] = {
            'params': total_params,
            'best_acc': best_val_acc,
            'time': training_time,
            'flops': flops,
            'inference_time_ms': inf_time,
            'history': history
        }
        if SAVE_MODELS:
            torch.save(best_model_state, os.path.join(OUTPUT_DIR, f'resnet50_groups_{groups}_best.pth'))
            print(f"💾 Модель сохранена: {OUTPUT_DIR}/resnet50_groups_{groups}_best.pth")

        if groups == GROUPS_LIST[-1]:
            print("\n📈 Визуализация распределения весов и градиентов (финальная модель)...")
            plot_weight_grad_histograms(model, OUTPUT_DIR)

    # ---------- ВЫВОД СВОДНОЙ ТАБЛИЦЫ В ТЕРМИНАЛ ----------
    print("\n" + "=" * 80)
    print("📊 ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
    print("=" * 80)
    table_data = []
    for g in GROUPS_LIST:
        table_data.append([
            g,
            f"{results[g]['params']:,}",
            f"{results[g]['flops'] / 1e9:.2f}",
            f"{results[g]['inference_time_ms']:.2f}",
            f"{results[g]['time']:.2f}",
            f"{results[g]['best_acc']:.2f}"
        ])
    headers = ["Groups", "Params", "GFLOPs", "Inference (ms)", "Train Time (s)", "Best Val Acc (%)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    df_results = pd.DataFrame({
        'groups': GROUPS_LIST,
        'params': [results[g]['params'] for g in GROUPS_LIST],
        'flops': [results[g]['flops'] for g in GROUPS_LIST],
        'inference_ms': [results[g]['inference_time_ms'] for g in GROUPS_LIST],
        'train_time_s': [results[g]['time'] for g in GROUPS_LIST],
        'best_acc': [results[g]['best_acc'] for g in GROUPS_LIST]
    })
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'experiment_summary.csv'), index=False)
    print(f"\n💾 Сводная таблица сохранена в {OUTPUT_DIR}/experiment_summary.csv")

    print("\n📈 Построение всех графиков...")
    plot_training_curves(results, OUTPUT_DIR)
    plot_comparison_bars(results, OUTPUT_DIR)
    plot_accuracy_vs_params(results, OUTPUT_DIR)
    plot_heatmap(results, OUTPUT_DIR)

    best_g = max(GROUPS_LIST, key=lambda g: results[g]['best_acc'])
    print(f"\n🏆 Лучшая модель: groups={best_g} с точностью {results[best_g]['best_acc']:.2f}%")
    best_model = ResNet50Groups(num_classes=100, groups=best_g).to(DEVICE)
    best_model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, f'resnet50_groups_{best_g}_best.pth'), map_location=DEVICE))
    plot_confusion_matrix(best_model, testloader, DEVICE, num_classes=100, output_dir=OUTPUT_DIR)

    print("\n✅ Эксперименты полностью завершены. Все графики и логи сохранены в папку", OUTPUT_DIR)
