import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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

GROUPS_LIST = [1, 2, 4, 8]
SAVE_MODELS = True
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

print("Загрузка CIFAR-100...")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
class_names = testset.classes  # список из 100 названий классов

# ---------------------- МОДЕЛЬ ResNet50Groups -------------------------
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

def compute_flops_approx(model):
    params = count_parameters(model)
    base_flops = 4.1e9
    base_params = 23_000_000
    flops = base_flops * (params / base_params) * 0.8
    return flops

try:
    from ptflops import get_model_complexity_info
    def compute_flops_precise(model, device):
        model_copy = ResNet50Groups(num_classes=100, groups=model.groups).to(device)
        macs, params = get_model_complexity_info(model_copy, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
        flops = macs * 2
        return flops, params
    USE_PTFlOPS = True
    print("✅ Используется ptflops для точного подсчёта FLOPs")
except ImportError:
    USE_PTFlOPS = False
    print("⚠️ ptflops не установлен. Будут использованы приблизительные FLOPs.")

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
    return np.mean(times) * 1000

def validate_full(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Валидация", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    acc = 100. * correct / total
    return acc, all_preds, all_targets

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:2d} Train", leave=False)
    for inputs, targets in pbar:
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
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
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

# -------------------------- ФУНКЦИИ ГРАФИКОВ --------------------------
def plot_comparison_bars(results_dict, output_dir):
    groups = list(results_dict.keys())
    params = [results_dict[g]['params'] / 1e6 for g in groups]
    best_acc = [results_dict[g]['best_acc'] for g in groups]
    train_time = [results_dict[g]['time'] for g in groups]
    flops = [results_dict[g]['flops'] / 1e9 for g in groups]
    inf_time = [results_dict[g]['inference_time_ms'] for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].bar([str(g) for g in groups], params, color='skyblue')
    axes[0,0].set_xlabel('Groups')
    axes[0,0].set_ylabel('Parameters (millions)')
    axes[0,0].set_title('Количество параметров')
    axes[0,0].grid(axis='y')

    axes[0,1].bar([str(g) for g in groups], best_acc, color='lightcoral')
    axes[0,1].set_xlabel('Groups')
    axes[0,1].set_ylabel('Test Accuracy (%)')
    axes[0,1].set_title('Точность на тесте')
    axes[0,1].grid(axis='y')

    axes[1,0].bar([str(g) for g in groups], train_time, color='lightgreen')
    axes[1,0].set_xlabel('Groups')
    axes[1,0].set_ylabel('Training Time (seconds)')
    axes[1,0].set_title('Время обучения (30 эпох)')
    axes[1,0].grid(axis='y')

    axes[1,1].bar([str(g) for g in groups], flops, color='gold')
    axes[1,1].set_xlabel('Groups')
    axes[1,1].set_ylabel('GFLOPs')
    axes[1,1].set_title('Вычислительная сложность')
    axes[1,1].grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bars.png'), dpi=150)
    plt.show()

def plot_accuracy_vs_params(results_dict, output_dir):
    groups = list(results_dict.keys())
    params = [results_dict[g]['params'] / 1e6 for g in groups]
    acc = [results_dict[g]['best_acc'] for g in groups]
    plt.figure(figsize=(8, 6))
    plt.scatter(params, acc, s=100, c=groups, cmap='viridis')
    for i, g in enumerate(groups):
        plt.annotate(f'groups={g}', (params[i], acc[i]), xytext=(5,5), textcoords='offset points')
    z = np.polyfit(params, acc, 1)
    p = np.poly1d(z)
    plt.plot(params, p(params), "r--", alpha=0.6, label=f'тренд (slope={z[0]:.2f})')
    plt.xlabel('Parameters (millions)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Зависимость точности от числа параметров')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_params.png'), dpi=150)
    plt.show()

def plot_heatmap(results_dict, output_dir):
    groups = list(results_dict.keys())
    df = pd.DataFrame({
        'Groups': groups,
        'Params_M': [results_dict[g]['params']/1e6 for g in groups],
        'Best_Acc': [results_dict[g]['best_acc'] for g in groups],
        'Time_s': [results_dict[g]['time'] for g in groups],
        'FLOPs_G': [results_dict[g]['flops']/1e9 for g in groups],
        'Inf_ms': [results_dict[g]['inference_time_ms'] for g in groups]
    }).set_index('Groups')
    corr = df.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Корреляция между метриками')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.show()

def plot_confusion_matrix_for_model(model, loader, device, groups, output_dir):
    acc, all_preds, all_targets = validate_full(model, loader, device)
    cm = confusion_matrix(all_targets, all_preds, labels=range(20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(20))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix (first 20 classes), groups={groups}, Acc={acc:.2f}%')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_groups_{groups}.png'), dpi=150)
    plt.show()

def plot_weight_histograms(model, groups, output_dir):
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weights.append(param.data.cpu().numpy().flatten())
    if weights:
        all_weights = np.concatenate(weights)
        plt.figure(figsize=(10, 4))
        plt.hist(all_weights, bins=50, alpha=0.7, label='Weights')
        plt.title(f'Гистограмма весов модели (groups={groups})')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'weights_hist_groups_{groups}.png'), dpi=150)
        plt.show()

# ---------- НОВАЯ ФУНКЦИЯ: сравнение предсказаний всех моделей ----------
def visualize_all_predictions(models_dict, loader, device, class_names, num_images=8, output_dir=None):
    """
    models_dict: словарь {groups: model} для всех загруженных моделей
    Показывает для каждого изображения: истинный класс и предсказание каждой модели с ✅/❌
    """
    images, targets = next(iter(loader))
    images = images[:num_images].to(device)
    targets = targets[:num_images].cpu().numpy()
    target_names = [class_names[t] for t in targets]

    predictions = {}
    for groups, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, preds = outputs.max(1)
        predictions[groups] = preds.cpu().numpy()

    fig, axes = plt.subplots(num_images, 1 + len(models_dict), figsize=(15, 3*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    for i in range(num_images):
        # Изображение
        img = images[i].cpu().numpy().transpose((1,2,0))
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(f"Image {i+1}", fontsize=10)
        axes[i, 0].set_title(f"True: {target_names[i]}")
        axes[i, 0].axis('off')

        for j, (groups, preds) in enumerate(predictions.items()):
            pred_class = class_names[preds[i]]
            is_correct = (preds[i] == targets[i])
            color = 'green' if is_correct else 'red'
            axes[i, j+1].text(0.5, 0.5, f"groups={groups}\n{pred_class}",
                              transform=axes[i, j+1].transAxes,
                              ha='center', va='center', fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            axes[i, j+1].axis('off')
            if i == 0:
                axes[i, j+1].set_title(f"Model groups={groups}")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'all_models_predictions.png'), dpi=150)
    plt.show()

# ----------------------------- ОБУЧЕНИЕ МОДЕЛИ -------------------------
def train_model(groups):
    print(f"\n🔥 Обучение модели с groups={groups}")
    model = ResNet50Groups(num_classes=100, groups=groups).to(DEVICE)
    total_params = count_parameters(model)
    print(f"📊 Параметров: {total_params:,}")
    flops = compute_flops(model, DEVICE)
    print(f"⚙️  FLOPs: {flops/1e9:.2f} GFLOPs")

    model_cpu = ResNet50Groups(num_classes=100, groups=groups).to('cpu')
    loader_cpu = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    inf_time = measure_inference_time(model_cpu, loader_cpu, 'cpu', num_batches=50)
    print(f"⏱️  Инференс (CPU, batch=1): {inf_time:.2f} ms")
    del model_cpu

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_acc = 0.0
    best_state = None
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, DEVICE, epoch)
        val_loss, val_acc = validate(model, testloader, criterion, DEVICE)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
        print(f"✅ Эпоха {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
    train_time = time.time() - start_time
    print(f"\n🏁 Обучение groups={groups} завершено за {train_time:.2f} сек. Лучшая точность: {best_acc:.2f}%")
    torch.save(best_state, os.path.join(OUTPUT_DIR, f'resnet50_groups_{groups}_best.pth'))
    print(f"💾 Модель сохранена: {OUTPUT_DIR}/resnet50_groups_{groups}_best.pth")

    model.load_state_dict(best_state)
    test_acc, _, _ = validate_full(model, testloader, DEVICE)
    print(f"📈 Точность на тестовой выборке: {test_acc:.2f}%")
    return best_acc, train_time, flops, inf_time, total_params, model

# ----------------------------- АНАЛИЗ ГОТОВЫХ МОДЕЛЕЙ ------------------
def analyze_saved_models(groups_list):
    print("\n" + "="*80)
    print("📊 АНАЛИЗ СОХРАНЁННЫХ МОДЕЛЕЙ")
    print("="*80)
    results = {}
    models_for_vis = {}
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    vis_loader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)  # для визуализации

    for groups in groups_list:
        model_path = os.path.join(OUTPUT_DIR, f'resnet50_groups_{groups}_best.pth')
        if not os.path.exists(model_path):
            print(f"⚠️ Модель для groups={groups} не найдена, пропускаем.")
            continue
        print(f"\n🔍 Загрузка модели groups={groups}")
        model = ResNet50Groups(num_classes=100, groups=groups).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        total_params = count_parameters(model)
        flops = compute_flops(model, DEVICE)
        model_cpu = ResNet50Groups(num_classes=100, groups=groups).to('cpu')
        loader_cpu = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        inf_time = measure_inference_time(model_cpu, loader_cpu, 'cpu', num_batches=50)
        del model_cpu
        print(f"📊 Параметров: {total_params:,}")
        print(f"⚙️  FLOPs: {flops/1e9:.2f} GFLOPs")
        print(f"⏱️  Инференс (CPU, batch=1): {inf_time:.2f} ms")

        test_acc, preds, targets = validate_full(model, testloader, DEVICE)
        print(f"🎯 Точность на тестовой выборке: {test_acc:.2f}%")

        results[groups] = {
            'params': total_params,
            'best_acc': test_acc,
            'time': 0,
            'flops': flops,
            'inference_time_ms': inf_time,
            'model': model,
            'preds': preds,
            'targets': targets
        }
        models_for_vis[groups] = model

    if not results:
        print("Нет загруженных моделей. Завершение.")
        return

    print("\n📈 Построение стандартных графиков...")
    plot_comparison_bars(results, OUTPUT_DIR)
    plot_accuracy_vs_params(results, OUTPUT_DIR)
    plot_heatmap(results, OUTPUT_DIR)

    for groups, data in results.items():
        plot_confusion_matrix_for_model(data['model'], testloader, DEVICE, groups, OUTPUT_DIR)
        plot_weight_histograms(data['model'], groups, OUTPUT_DIR)

    # Визуализация сравнения предсказаний всех моделей
    if len(models_for_vis) > 0:
        print("\n🖼️  Визуализация сравнения предсказаний всех моделей...")
        visualize_all_predictions(models_for_vis, vis_loader, DEVICE, class_names, num_images=8, output_dir=OUTPUT_DIR)

    print("\n" + "="*80)
    print("📋 СВОДНАЯ ТАБЛИЦА ПО МОДЕЛЯМ")
    print("="*80)
    table_data = []
    for g, data in results.items():
        table_data.append([
            g,
            f"{data['params']:,}",
            f"{data['flops']/1e9:.2f}",
            f"{data['inference_time_ms']:.2f}",
            f"{data['best_acc']:.2f}"
        ])
    headers = ["Groups", "Params", "GFLOPs", "Inference (ms)", "Test Acc (%)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    best_g = max(results.keys(), key=lambda g: results[g]['best_acc'])
    print(f"\n🏆 Лучшая модель: groups={best_g} с точностью {results[best_g]['best_acc']:.2f}%")

    df = pd.DataFrame([{
        'groups': g,
        'params': data['params'],
        'flops': data['flops'],
        'inference_ms': data['inference_time_ms'],
        'test_acc': data['best_acc']
    } for g, data in results.items()])
    df.to_csv(os.path.join(OUTPUT_DIR, 'analysis_summary.csv'), index=False)
    print(f"\n💾 Результаты анализа сохранены в {OUTPUT_DIR}/analysis_summary.csv")
    print("✅ Анализ завершён.\n")

# ----------------------------- ОСНОВНАЯ ПРОГРАММА С МЕНЮ -----------------
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    print("\n" + "="*60)
    print("   ВЫБОР ДЕЙСТВИЯ ДЛЯ КУРСОВОЙ РАБОТЫ (Тема №10)")
    print("="*60)
    print("1️⃣  Обучить модель для выбранной группы (1,2,4,8)")
    print("2️⃣  Проанализировать уже обученные модели (по сохранённым .pth)")
    print("="*60)

    choice = input("Ваш выбор (1 или 2): ").strip()

    if choice == '1':
        print("\nДоступные группы: 1, 2, 4, 8")
        grp = input("Введите номер группы для обучения: ").strip()
        if grp not in ['1','2','4','8']:
            print("Неверный ввод. Завершение.")
            sys.exit(0)
        groups = int(grp)
        model_path = os.path.join(OUTPUT_DIR, f'resnet50_groups_{groups}_best.pth')
        if os.path.exists(model_path):
            overwrite = input(f"Модель для groups={groups} уже существует. Переобучить? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Обучение отменено.")
                sys.exit(0)
        train_model(groups)

    elif choice == '2':
        existing_groups = []
        for g in [1,2,4,8]:
            if os.path.exists(os.path.join(OUTPUT_DIR, f'resnet50_groups_{g}_best.pth')):
                existing_groups.append(g)
        if not existing_groups:
            print("❌ Не найдено ни одной сохранённой модели. Сначала обучите хотя бы одну модель (пункт 1).")
            sys.exit(0)
        print(f"\n🔍 Найдены модели для groups: {existing_groups}")
        analyze_saved_models(existing_groups)
    else:
        print("Неверный выбор. Завершение.")
