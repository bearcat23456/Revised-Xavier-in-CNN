import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Device Configuration and Data Loading ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)


# --- 2. ResNet18 Model Definition ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# --- 3. Initialization Schemes Definition ---
def weights_init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None: m.bias.data.zero_()


def weights_init_revised_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1] if isinstance(m, nn.Conv2d) else m.in_features
        fan_out = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
        a = np.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(m.weight.data, -a, a)
        if m.bias is not None: m.bias.data.zero_()


def weights_init_he(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None: m.bias.data.zero_()


def weights_init_uniform(m, val):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight.data, -val, val)
        if m.bias is not None: m.bias.data.zero_()


# --- 4. Training and Evaluation Function ---
def train_and_evaluate(model, trainloader, optimizer, criterion):
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Calculate final accuracy after one epoch
    model.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    return 100 * correct_train / total_train


# --- 5. Run Experiments ---
def run_experiment():
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    final_accuracies = {}

    # Define all initialization schemes to be tested
    init_schemes = {
        'Standard Xavier': lambda m: weights_init_xavier(m),
        'Revised Xavier': lambda m: weights_init_revised_xavier(m),
        'He Initialization': lambda m: weights_init_he(m),
    }
    uniform_vals = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 10]

    # Train and log accuracy for special initialization schemes
    special_init_data = {}
    for name, init_func in init_schemes.items():
        print(f"--- Running experiment for: {name} ---")
        model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
        model.apply(init_func)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        accuracy = train_and_evaluate(model, trainloader, optimizer, criterion)
        special_init_data[name] = accuracy
        print(f"Final training accuracy for {name}: {accuracy:.2f}%\n")

    # Train and log accuracy for a range of uniform distributions
    uniform_data = {}
    for val in uniform_vals:
        name = f'Uniform +/-{val}'
        print(f"--- Running experiment for: {name} ---")
        model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
        model.apply(lambda m: weights_init_uniform(m, val))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        accuracy = train_and_evaluate(model, trainloader, optimizer, criterion)
        uniform_data[name] = accuracy
        print(f"Final training accuracy for {name}: {accuracy:.2f}%\n")

    final_accuracies.update(special_init_data)
    final_accuracies.update(uniform_data)
    return final_accuracies


if __name__ == '__main__':
    all_results = run_experiment()

    # --- 6. Results Visualization and Data Output ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))

    # Separate results into uniform and special initializations
    uniform_results = {float(key.split('+/-')[1]): value for key, value in all_results.items() if
                       key.startswith('Uniform')}
    special_results = {key: value for key, value in all_results.items() if not key.startswith('Uniform')}

    # Plot uniform distribution curve
    sorted_uniform_vals = sorted(uniform_results.keys())
    sorted_uniform_accs = [uniform_results[val] for val in sorted_uniform_vals]

    plt.plot(range(len(sorted_uniform_vals)), sorted_uniform_accs,
             marker='o', linestyle='-', color='blue', linewidth=4,
             label='Uniform Distribution Range')

    # Plot special initialization points
    special_labels = list(special_results.keys())
    special_accs = list(special_results.values())
    markers = ['s', '^', 'D']  # Square, Triangle, Diamond
    colors = ['red', 'green', 'purple']

    # Place special initialization points on the graph with different shapes and colors
    special_x_positions = [len(sorted_uniform_vals) + i for i in range(len(special_labels))]
    for i in range(len(special_labels)):
        plt.scatter(special_x_positions[i], special_accs[i],
                    marker=markers[i], s=200, color=colors[i],
                    label=special_labels[i], zorder=10)  # zorder ensures points are on top of the line

    plt.title('Final Training Accuracy by Initialization Method (1 Epoch)', fontsize=16)
    plt.xlabel('Initialization Method', fontsize=14)
    plt.ylabel('Final Training Accuracy (%)', fontsize=14)

    # Set x-axis labels
    all_x_labels = [f'+/-{val}' for val in sorted_uniform_vals] + special_labels
    all_x_ticks = list(range(len(all_x_labels)))
    plt.xticks(all_x_ticks, all_x_labels, rotation=45, ha='right', fontsize=10)

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print("\n--- Experiment Data Log ---")
    for name, acc in all_results.items():
        print(f"'{name}': {acc:.2f}%")