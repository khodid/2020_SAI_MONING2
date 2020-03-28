```python

# Package
import torch
import torch.optim as optim

# minibatch 생성하기

from torch.utils.data import Dataset
# 원하는 Dataset을 지정할 수 있게 됨

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152],[185],[180],[196],[142]]
    # __len__() : 이 데이터셋의 총 데이터수
    def __len__(self):
        return len(self.x_data)
    # __getitem__() :  index를 받았을 때 그에 상응하는 입출력 데이터 반환
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x, y

dataset = CustomDataset()

from torch.utils.data import DataLoader

dataloader = DataLoader(
	  dataset,		
    batch_size = 2,		# Size of each minibatch
    shuffle = True		# 프로그램이 순서 자체를 학습할 위험이 있어, Batch 생성시마다 순서를 바꿔줌.
)

# lab 04 -1 실습 코드 중 클래스 선언하는 쪽 이용.
import torch.nn as nn
import torch.nn.functional as F

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self, x):
        return self.linear(x)

# Model
model = MultivariateLinearRegressionModel()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader): 
		    # enumerate(dataloader):    minibatch 인덱스와 데이터를 받음
        x_train, y_train = samples
        
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Batch{}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))

```
