```python

# Package

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Torch Seed 부여
torch.manual_seed(1)

# Training Data Set
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]  # |x_data| = 6X2
y_data = [[0], [0], [0], [1], [1], [1]]			# |y_data| = 6X1

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Hypothesis
W = torch.zeros((2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# Optimizer 설정
optimizer = optim.SGD([W,b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # Cost 계산

    # 1 - 수식 그대로 표현
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    # 2 - torch 제공함수로 계산
    # hypothesis = torch.sigmoid(x_train.matmul(W)+ b)
    
    # 1 - 수식 그대로 표현
    losses = -(y_train[0]*torch.log(hypothesis[0]) + 
		(1-y_train) * torch.log(1-hypothesis[0]))
    cost = losses.mean()

    # 2 - torch 제공함수로 계산 : BCE(Binary Cross Entropy)
    # cost = F.binary_cross_entropy(hypothesis, y_train)
    
    # Calculate H(x)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력하게
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch,nb_epochs, cost.item()))

```
