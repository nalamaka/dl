import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR = 0.001
EPOCH = 5

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root="../../data/",
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root="../../data/",
    train=False,
    transform=transform
)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
def train_model(cnn):
    optimizer = optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        cnn.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.to(device), b_y.to(device)

            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                cnn.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for b_x_test, b_y_test in test_loader:
                        b_x_test = b_x_test.to(device)
                        b_y_test = b_y_test.to(device)

                        output = cnn(b_x_test)
                        pred_y = torch.max(output, 1)[1]

                        correct += (pred_y == b_y_test).sum().item()
                        total += b_y_test.size(0)

                accuracy = correct / total
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
                cnn.train()
                
        
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU is available. Using CUDA.")
        device = torch.device("cuda")
    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")
    cnn = CNN()
    cnn.to(device)
    train_model(cnn)