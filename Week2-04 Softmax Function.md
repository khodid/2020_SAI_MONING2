# Softmax Regression

Logistic Classification을 통해서 Binary Classification을 한다는 걸 알았으니,
이제 Multinomial Classification의 경우를 생각 해보자.

## 이론

- **개념**

  만약 data를 A, B, C로 나누어야 하는 상황이라면,
  A, B에 속하는 데이터를 'C가 아님'으로 생각하고, C에 속하는 데이터를 'C임' 으로 생각하면
  Binary Classification과 동일하게 된다.

  ![](https://miro.medium.com/max/608/1*SwXHlCzh-d9UqHOglp3vcA.png)

  따라서 모델은 'A or not?', 'B or not?', 'C or not?' 세가지 모델을 쓰는게 된다.

  즉, 동일한 X 벡터를 세 가지 W 벡터에 대해 연산을 하게 되는 건데,
  이 건 행렬곱의 원리를 이용해 하나의 연산으로 합칠 수가 있다.

  참고 : 

  ![](https://t1.daumcdn.net/cfile/tistory/2425554058D226BC27)
  ![](https://t1.daumcdn.net/cfile/tistory/213ED74158D227830A)

 

- Softmax의 필요성

  - 0과 1 사이의 값이 나왔으면 좋겠다.
  - A에 대한, B에 대한, C에 대한 y output의 Sum이 1이 되었으면 좋겠다
    (꼭 확률처럼)

  과 같은, Sigmoid를 썼을 때와 비슷한 이유로 softmax가 필요하다.

- Cost function : **Cross - Entropy**

  예측 값과 실제 값의 차이가 얼마인지 구하는 cost function까지 완성되어야 함!
  일단 생김새는 이렇다.
  $$
  D(S, L) = -\sum_i L_i\log(S_i)
  $$
  [이게 왜 설득력 있는 cost function이 되는가?(강의영상)](https://youtu.be/jMU9G5WEtBc?t=314)
  요약하자면 맞으면 그냥 넘어가고 하나 틀리면 어마무시하게 값이 커지는 뭐 그런 거라고 한다.

- Logistic cost VS cross entropy

  - Logistic Cost
    $$
    C:(H(x),y) = y\log(H(x)) - (1-y)\log(1-H(x))
    $$

  이 것이 사실상 Cross Entropy와 같다.
  H(x) == S, y ==L 이렇게 등치됨.

- Cost function (full version)
  $$
  \mathcal{L} = \frac{1}{N} \sum_i D(S(wx_i+b)L_i)
  $$
  이렇게 다 더해서 평균을 하는 것까지 하면 Cost Function의 완성이다.

  - Minimalize (Gradient Descent)

    경사면을 미분하는 것은 다루지 않겠다. 암튼 gradient 값이 최소값을 향하게 해준다는 걸 알면 된다...

    

## 실습

- Settings

  - Package

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    ```

  - For reproducibility / 같은 결과를 보장하기 위해서  -?

    ```python
    torch.manual_seed(1)
    ```

- Softmax 함수란?

  - 개념 

     [1, 2, 3]을 일반적인 max에 넣는다면, [0, 0, 1]이 결과로 나올 것이다.
    하지만 softmax의 경우 좀더 명확하지 않은 '비율'로 max를 계산한다.
    앞선 예시와 같이 [1, 2, 3]을 넣었다면 softmax의 결과는 [0.0900, 0.2447, 0.6652] 뭐 이런 식

    셋을 다 합치면 1이 나오니, 어떻게 보면 이산확률분포처럼 생각할 수 있다.

  - 형태
    $$
    P(\mbox {class = i}) = \frac{e^i}{\sum e^i}
    $$

  - Code

    ```python
    hypothesis = F.softmax(z, dim=0)
    ```

    

- Cross Entropy

  - 수식
    $$
    H(P,Q) = - \mathbb E_{x\sim P(x)}[logQ(x)] = -\sum_{x\in X} P(x)\log(Q(x))
    $$

  -  :: 설명이 있긴 한데 이해가 안 간다... 추가 조사가 필요함

  - Cross Entropy Loss __(Low - Level)__
    $$
    L = -\frac{1}{N} \sum -y \log(\hat y)
    \\
    \hat y \mbox{은 예측한 확률값,  } y \mbox{는 실제 확률값(0 또는 1)}
    $$
    

    ``` python
    z = torch.rand(3, 5, requires_grad = True)	
    # softmax의 예시로 그냥 랜덤한 수.
    # 3X5니깐 class는 5개, sample은 3개
    hypothesis = F.softmax(z, dim = 1)			
    # dim=1: --> 방향으로 softmax를 하라.
    
    # 예를 드는 거니깐 정답도 임의로 생성하자.
    y = torch.randint(5,(3,)).long()			
    
    y_one_hot = torch.zeros_like(hyothesis) 	# H와 같은 크기(3,5)의 텐서 선언
    y_one_hot.scatter_(1,y.unsqueeze(1),1)		
    # 첫 parameter는 dim을 나타내고, 마지막 parameter는 어떤 값을 넣을지 정하는 것.
    # 참고로 unsqueeze의 parameter도 dim. 명시한 dim 방향에 따라 차원 추가함. 3 -> (3,1)
    
    # 결과로 쉽게 말하자면, 원래 [0, 2, 1]이던 걸 이용해
    # [[1, 0, 0, 0, 0],
    #  [0, 0, 1, 0, 0],
    #  [0, 1, 0, 0, 0]]
    # 으로 바꾸는 뭐 그런 과정이다.
    # 암튼 이렇게 정답 벡터를 임의로 만들어냈다.
    
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    # |y_one_hot|== (3,5) and |hypothesis| == (3,5)
    # sum(dim=1)을 해주니 둘을 곱한 값 (3,5)를 -> 방향으로 합해줌. 결과로 (3,)
    # 그리고 그걸 평균냄 (3,) --> Scalar
    ```

  - PyTorch 제공 함수를 이용

    ``` python
    # Low level
    torch.log(F.softmax(z,dim=1))
    
    # High level
    F.log_softmax(z, dim=1)
    
    # 둘이 동일한 값을 낸다는 걸 이용해서 High level스럽게 수정을 해보자.
    
    #기존
    hypothesis = F.softmax(z, dim = 1)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    # 1차 수정
    cost = (y_one_hot * -F.log_softmax.sum(dim=1).mean()
    # 2차 수정
    cost = F.nll_loss(F.log_softmax(z , dim=1), y)
    # 3차 수정
    cost = F.cross_entropy(z, y)
    ```

    저기서 nll은 Negative Log Likelihood의 약자다.

    보통은, 특히 뉴럴 네트워크 이론에선 prediction 단계에서 확률값을 알아야할 때가 있어서,
    3차 수정에서 쓴 Cross Entropy 함수를 쓰면 곤란할 수 있다.

    따라서 사용자의 판단에 따라서 알맞은 것으로 쓰면 되겠다.

- Training

  - Training Set

    ```python
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
    y_train = torch.FloatTensor(y_train)
    ```

  - 모델 초기화

    ``` python
    W = torch.zeros((4,3), requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    ```

  - Optimizer 설정

    ```python
    optimizer = optim.SGD([W,b], lr = 0.1)
    ```

  - 학습 (1) - Low Level

    ```python
    nb_epochs = 1000
    for epoch in range(nb_epochs+1):
        
        # cost 계산
        hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
        y_one_hot = torch.zeros_like(hyothesis)
    	y_one_hot.scatter_(1, y_train.unsqueeze(1),1)	
        cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
        
        # H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost : {:6f}'.format(epoch, nb_epochs, cost.item()))
        
    ```

  - 학습 (2) - with F.cross_entropy

    ```python
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
    ```

  

## 실전 

High level Implementation with nn.Module



```python
# 클래스 정의
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)   # 4개의 확률값을 받아 3개의 vector 생성
        
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()		# 선언

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr = 0.1)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(X) 계산
    prediction = model(x_train)   # |x_train| = (m,4), |prediction| = (m,3)
    
    # cost 계산
    cost = F.cross_entropy(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{}  Cost: {:.6f}'.format(epoch), nb_epochs, cost.item())
        
```



## 강조

Binary Classification의 경우: 

​	Binary Cross Entropy (BCE)
​	Sigmoid

Multinomial Classification의 경우:

​	Cross Entropy(CE)
​	Softmax

를 쓰는 걸 실전에서 주의하도록 하라.