import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 128
hidden_layer = 500
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.activation(x1)
        out = self.fc2(x2)
        return out


net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
losses = []
# Train the Model
for epoch in range(num_epochs):
    total_loss = 0
    print(epoch)
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        probs = net(images)
        loss = criterion(probs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(total_loss / len(train_loader))


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    net.eval()
    probs = net.forward(images)
    total += labels.size(0)
    predicted = torch.argmax(probs.data, 1)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
# torch.save(net.state_dict(), 'model.pkl')
import matplotlib.pyplot as plt

plt.plot(range(num_epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Per Epoch")
plt.show()
