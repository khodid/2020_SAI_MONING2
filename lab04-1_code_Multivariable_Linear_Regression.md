```python

# 0. Package import
import torch
import torch.optim as optim

# 1. Training Data Initialize
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# 2. Model Initialization
W = torch.zeros((3,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# 3. Optimizer
optimizer = optim.SGD([W,b], lr=1e-5)

nb_epochs = 20                  # 학습 횟수
for epoch in range(nb_epochs + 1):
    # 4. Hypothesis 
    hypothesis = x_train.matmul(W) + b # or .mm or @
    # 5. Cost 계산
    cost = torch.mean((hypothesis - y_train)**2)
    # 6. Gradient Descent
    optimizer.zero_grad()       # gradient 0으로 초기화
    cost.backward()             # cost function 미분 - gradient 계산
    optimizer.step()            # Gradient Descent 수행
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))
    
 ```
