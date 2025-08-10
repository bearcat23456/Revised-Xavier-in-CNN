import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for convenient data handling

# Ensure CUDA is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading and Preprocessing ---
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# --- 2. Model Definition (ResNet18) ---
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


# --- 3. Initialization Methods ---
def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_my_xavier(m):
    if isinstance(m, nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        fan_out = m.out_channels

        std = (2.0 / (fan_in + fan_out)) ** 0.5
        a = (3.0) ** 0.5 * std

        nn.init.uniform_(m.weight.data, -a, a)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


# --- 4. Training and Evaluation Functions ---
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(model, trainloader, testloader, optimizer, criterion, epochs, sampling_interval):
    model.train()
    batch_counter = 0
    train_accuracies = []
    sampled_batches = []

    # Used to calculate training accuracy for the current sampling interval
    correct_in_interval = 0
    total_in_interval = 0

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Record training accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)
            total_in_interval += labels.size(0)
            correct_in_interval += (predicted == labels).sum().item()

            batch_counter += 1
            if batch_counter % sampling_interval == 0:
                train_acc = 100 * correct_in_interval / total_in_interval
                train_accuracies.append(train_acc)
                sampled_batches.append(batch_counter)
                print(f'Batch [{batch_counter}] | Train Acc: {train_acc:.2f}%')

                # Reset counters
                correct_in_interval = 0
                total_in_interval = 0

    print('Finished Training')
    return train_accuracies, sampled_batches


# --- 5. Run Experiments and Comparison ---
def run_experiment():
    EPOCHS = 5
    learning_rate = 0.001
    SAMPLING_INTERVAL = 100

    # --- Experiment 1: Standard Xavier Initialization ---
    print("--- Running Experiment with Standard Xavier Initialization ---")
    resnet_standard = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    resnet_standard.apply(weights_init_xavier)

    criterion = nn.CrossEntropyLoss()
    optimizer_standard = optim.SGD(resnet_standard.parameters(), lr=learning_rate)

    train_acc_standard, batches = train_model(
        resnet_standard, trainloader, testloader, optimizer_standard, criterion, epochs=EPOCHS,
        sampling_interval=SAMPLING_INTERVAL
    )

    print("\n" + "=" * 50 + "\n")

    # --- Experiment 2: My Xavier Initialization ---
    print("--- Running Experiment with My Xavier Initialization ---")
    resnet_my = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    resnet_my.apply(weights_init_my_xavier)

    optimizer_my = optim.SGD(resnet_my.parameters(), lr=learning_rate)

    train_acc_my, _ = train_model(
        resnet_my, trainloader, testloader, optimizer_my, criterion, epochs=EPOCHS, sampling_interval=SAMPLING_INTERVAL
    )

    # --- 6. Results Visualization ---
    plt.figure(figsize=(15, 8))
    plt.plot(batches, train_acc_standard, label='Standard Xavier', linewidth=3)
    plt.plot(batches, train_acc_my, label='My Xavier (fan_out = C_out)', linewidth=3)
    plt.title('Training Accuracy Comparison on CIFAR-10', fontsize=16)
    plt.xlabel('Batches', fontsize=14)
    plt.ylabel('Training Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_experiment()