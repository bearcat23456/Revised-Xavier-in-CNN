import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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



class CustomConvNet(nn.Module):
    def __init__(self):
        super(CustomConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.tanh3 = nn.Tanh()

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.tanh3(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = self.tanh4(self.fc1(x))
        x = self.fc2(x)
        return x



def weights_init_standard_xavier(m):

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None: m.bias.data.zero_()


def weights_init_revised_xavier(m):

    if isinstance(m, nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        fan_out = m.out_channels
        std = np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        nn.init.uniform_(m.weight.data, -a, a)
        if m.bias is not None: m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_in = m.in_features
        fan_out = m.out_features
        std = np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        nn.init.uniform_(m.weight.data, -a, a)
        if m.bias is not None: m.bias.data.zero_()



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


def train_model(model, trainloader, testloader, optimizer, criterion, model_name, results_dict):
    model.train()
    batch_counter = 0
    test_accs = []

    for epoch in range(7):
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_counter += 1
            if batch_counter % 500 == 0:
                test_acc = evaluate_accuracy(model, testloader)
                test_accs.append(test_acc)
                print(f'Model: {model_name} | Epoch: {epoch + 1} | Batch: {batch_counter} | Test Acc: {test_acc:.2f}%')

    results_dict[model_name] = test_accs
    print(f'Finished training for {model_name}\n')



def run_experiment():
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    results = {}


    schemes = {
        'Standard Xavier': weights_init_standard_xavier,
        'Revised Xavier': weights_init_revised_xavier,
    }

    for name, init_func in schemes.items():
        print(f"--- Running experiment for: {name} ---")
        model = CustomConvNet().to(device)
        model.apply(init_func)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        train_model(model, trainloader, testloader, optimizer, criterion, name, results)

    return results


if __name__ == '__main__':
    all_results = run_experiment()


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    batch_intervals = [i * 500 for i in range(1, len(all_results['Standard Xavier']) + 1)]

    plt.plot(batch_intervals, all_results['Standard Xavier'], label='Standard Xavier',
             marker='o', linestyle='-', color='blue', linewidth=3)

    plt.plot(batch_intervals, all_results['Revised Xavier'], label='Revised Xavier',
             marker='^', linestyle='--', color='red', linewidth=3)

    plt.title('Test Accuracy Comparison: Standard vs. Revised Xavier', fontsize=16)
    plt.xlabel('Batches', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n--- Experiment Data Log ---")
    print("Logged Batches:", batch_intervals)
    for name, accs in all_results.items():
        print(f"\n{name}:")
        for i, acc in enumerate(accs):
            print(f"  Batch {batch_intervals[i]}: {acc:.2f}%")