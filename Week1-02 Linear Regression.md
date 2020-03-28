## Linear Regression ##

이론 말고 코딩하는 방법을 배울 것임.

예시로는 공부한 시간(Input; x)에 대비한 점수(Output; y)를 예측하는 프로그램.



### Data definition ###

``` python
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
```



### Hypothesis (Linear Regression) ###

$$
H(x) = Wx + b
$$



``` python
# Weight, Bias를 0으로 초기화 -> zeros
# requires_grad = True 로 해줌으로써 학습해나가야 한다는 것을 명시

# 참고 : Parameters of torch.zeros
# torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

w = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)
 
hypothesis = x_train * W + b
```



### Compute Loss ###

- Mean Squared Error(MSE)

$$
\frac{1}{m} \sum_{i=1}^m (H(x^{(i)} - y^{(i)}))^2
$$

​	 이걸 수식으로 구현하면 

``` python
cost = torch.mean((hypothesis - y_train) **2)
```

​	정도로 간단히 표현 가능하다.



###  Gradient Descent ###

- torch.optim 라이브러리 사용
  - parameter : [W, b] 는 학습할 tensor들, lr은 learning rate 

``` python

optimizer = optim.SGD([W,b], lr=0.01)

# 항상 붙어다니는 3줄
optimizer.zero_grad()	# gradient 초기화
cost.backward()			# gradient 계산
optimizer.step()		# W,b 개선

```

- 실전 코드

  ``` python
  # setup : 
  x_train = torch.FloatTensor([[1],[2],[3]])		
  y_train = torch.FloatTensor([[2],[4],[6]])			# Data 정의
  
  w = torch.zeros(1, requires_grad = True)			
  b = torch.zeros(1, requires_grad = True)			# Hypothesis 초기화
  
  optimizer = optim.SGD([W,b], lr=0.01)				# Optimizer 정의
  
  nb_epochs = 1000									# 학습 횟수
   # loop : 
  for epoch in range(1, nb_epochs + 1):
      hypothesis = x_train * W + b 					#Hypothesis 예측
  	cost = torch.mean((hypothesis - y_train) **2)	# Cost 계산 
  
      # 학습
  	optimizer.zero_grad()				# gradient 0으로 초기화
  	cost.backward()						# cost function 미분 - gradient 계산
  	optimizer.step()					# Gradient Descent 수행
  
  ```



### Why Gradient Descent(경사하강법)? ###

- 개인적으로 알고 있는 지식 : Gradient  Vector는 가장 기울기가 높은 방향을 가리킴. 각 축에 대해 편미분하고 다 더하면 됨. 근데 2차원이니깐 중심과 먼 쪽을 가리킬 것임...
  
  
  
- cost(W)를 그래프로 나타내면 대략 오목한 2차 함수가 나온다.
  
  이 2차 함수의 꼭지점(cost가 가장 적은 지점)을 기계적으로 찾아내는 것이 
  *경사하강법(Gradient Descent Algoritm)*이다.
  
  - 경사가 어떤지 보고 그 방향으로 W를 조금씩 움직이는 기술.
  
  
  
  
  $$
  \nabla W = {\partial cost\over\partial W} = \frac{2}{m} \sum_{i=1}^{m}(Wx^{(i)} - y^{(i)})x^{(i)}
  $$
  

$$
W : = W - \alpha \nabla W
$$

- 수식에서 alpha는 W를 얼마나 움직일지 정하는 상수, Learning rate(lr) 과 같음 

- 어떤 점에서 시작하든 간에 항상 최소점에 도달할 수 있다.
- 이 방법을 위해서 cost(W)에서 1/m 대신 1/2m로 정의하기도 한다. 
  (미분했을 때 상수를 없애기 위해)



- In code ...

  ``` python
  gradient = 2* torch.mean((W * x_train - y_train) * x_train)
  lr = 0.1
  W -= lr*gradient
  ```

  

- Full Code (Without 'optim' Funtion)

  ``` python
  x_train = torch.FloatTensor([[1],[2],[3]])
  y_train = torch.FloatTensor([[1],[2],[3]])
  
  W = torch.zeros(1)
  lr = 0.1
  
  nb_epoches = 10
  for epoch in range(nb_epoches + 1):
      htpothesis = x_train * W
      
      cost = torch.mean((hypothesis - x_train) **2)
      gradient = torch.sum((W * x_train - y_train) * x_train)
      
      print('Epoch {:4d}/{} W: {:.3f}, Cost {:.6f}'.format(epoch, nb_epochs,W.item(),cost.item()))
      
      W -= lr*gradient
      
  ```

 

