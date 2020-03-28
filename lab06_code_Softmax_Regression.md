``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Training Set
x_train = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]					# |x_train| = (m, 4) 
										#  즉 sample 개수 m, Class개수 4
y_train = [2, 2, 2, 1, 1, 1, 0, 0]		
# One Hot vector로 나타냈을 때 1이 있는 자리의 index
x_train = torch.FloatTensor(x_train)	# |y_train| = (m)	
y_train = torch.LongTensor(y_train)   # y_train이 Long형이어야 F.cross_entropy 함수의 parameter로 들어갈 수 있음.


# 모델 설정
W = torch.zeros((4,3), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# Optimizer 설정
optimizer = optim.SGD([W,b], lr = 0.1)

# 학습 1 - Low level
#"""
nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    # cost 계산
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1),1)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    
    # H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost : {:6f}'.format(epoch, nb_epochs, cost.item()))
#"""

# 학습 2 - with F.cross_entropy
"""
nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    # cost 계산
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train) 
    # One hot vector 만드는 과정이 생략된 걸 볼 수 있다
    
    # H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost : {:6f}'.format(epoch, nb_epochs, cost.item()))

"""

```
