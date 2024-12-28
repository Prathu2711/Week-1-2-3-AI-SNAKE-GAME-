import torch 
import torch.nn as nn
import torchvision #for datasets
import torchvision.transforms as transforms
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #checking cuda support

#HYPER PARAMETERS
input_size = 784 #28*28 that is image size
hidden_size = 100
num_classes = 10 #there are 10 digits
num_epochs = 3
batch_size = 10
learning_rate = 0.001

#IMPORTING MNIST DATA
train_dataset = torchvision.datasets.MNIST(root = './data', train = True , download = True , transform = transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root = './data', train = False , transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset ,batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset ,batch_size = batch_size,shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x , dim = 1)

net = Net()
#OPTIMIZERS AND LOSS  
import torch.optim as optim
optimizer = optim.Adam(net.parameters(),lr = 0.001)

for epochs in range(num_epochs):
    for i,data in enumerate(train_loader):
        X,y = data
        net.zero_grad()
        output = net(X.view(-1,28*28))
        loss = F.nll_loss(output,y)
        loss.backward()
        optimizer.step()

        if((i+1) % 100 == 0):
            print(f'epoch {epochs+1} / {num_epochs}, step {i+1}/{len(train_loader)}, loss = {loss.item():.4f}')

correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        X,y = data
        output = net(X.view(-1,784))
        for idx,i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total,3))


