## Multivariable Linear Regression ##

저번까진 하나의 정보만으로 예측을 하는 것을 배웠는데, 이번에는 복수의 정보를 기반으로 하나의 추측값을 도출하는 다항 선형 회귀를 해볼 것이다.



- Data (example)

  ``` python
  x_train = torch.FloatTensor([[73, 80 75],
                              [93, 88, 93],
                              [89, 91, 90]
                              [96, 98, 100]
                              [73, 66, 70]])
  y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])
  
  ```

  

- Hypothesis Function
  $$
  H(x) = w_1x_1 + w_2x_2 + w_3x_3 + b
  $$
  

  Data input의 가지 수에 따라 w의 수도 똑같이 맞춰주는 게 인지상정.

  - In code...

    ```python
    # Since there are so many arguments...
    # We will use 'matmul()'
    hypothesis = x_train.matmul(W) + b
    ```

    간결할 뿐만 아니라 더욱 빠르기도 하다고 함.

  

- Cost function : MSE

  - Same as Simple Linear Regression

    ``` python
    cost = torch.mean((hypothesis - y_train) **2)
    ```

    

- Gradient Descent with torch.optim

  - Same as Simple Linear Regression

  

### Full Code Example with torch.optim

``` python
import torch
import torch.optim as optim

# 1. Data Initialization
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

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # 4. Hypothesis 
    hypothesis = x_train.matmul(W) + b # or .mm or @
    # 5. Cost
    cost = torch.mean((hypothesis - y_train)**2)
    # Gradient Descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))
    
    
```

​	결과로는 점점 Cost가 작아지고 점점 y에 가까워지는 H(x)를 볼 수 있음

​	 lr 잘못 설정하면 발산할 수도 있음



### nn.Module

- **모델 초기화** 과정을 간편하게 만들기 위해 있는 모듈.

  ``` python
  # 2. Model Initialization
  W = torch.zeros((3,1), requires_grad = True)
  b = torch.zeros(1, requires_grad = True)
  
  # ...
  
  hypothesis = x_train.matmul(W) + b
  ```

  이 부분을

  ```python
  import torch.nn as nn
  class MultivariateLinearRegressionModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = nn.Linear(3,1)
          
      def forward(self, x):
          return self.linear(x)
      
  hypothesis = model(x_train)
  ```

  로 표현할 수도 있음.

  - nn.Module을 상속해서 모델 생성
  - nn.Linear(3,1) : (입력차원, 출력차원) 을 파라미터로 넣기
  - Hypothesis 계산은 forward 함수에 어떻게 하는지만 알려주기
  - Gradient 계산은 PyTorch에서 알아서 해줌 backward()



### PyTorch 제공 Cost Funtion

- 왜 쓸까?

  다른 Cost Function으로 전환할 때 편리함

  계산 오류를 피할 수 있어 디버깅할 때 편리함

- Code

  ``` python
  import torch.nn.function as F
  
  cost = F.mse_loss(prediction, y_train)
  # 기존:  cost = torch.mean((hypothesis - y_train)**2)
  ```

- 제공되는 다른 cost funtion 예:

  - l1_loss

  - smooth_l1_loss

  - etc.

    

### Module 이용 Full Code

``` python
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

