# PyTorch: Tensor Manipulation

![1](https://res.cloudinary.com/practicaldev/image/fetch/s--oTgfo1EL--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/adhiraiyan/DeepLearningWithTF2.0/master/notebooks/figures/fig0201a.png)

- 0차원: 스칼라 (Scalar)
  1차원: 벡터 (Vector)
  2차원: 행렬 (Matrix)
  3차원: 텐서 (Tensor)
  4차원 이상은 상상하기 어렵다.

- Tensor의 크기?

  - 2D Tensor의 경우
    |t| = (batch size, dimension) 

  - 3D Tensor의 경우 - 1. Typical Computer Vision
    |t| = (batch size, width, height)

  - 3D Tensor의 경우 - 2. Typical Natural Language Processing (Sequential)

    |t| = (batch size, length, dimension)

- Import

  ``` python
  import numpy as np
  import torch
  ```

- 1D & 2D Array with NumPy 

  ``` python
  t1 = np.array([0., 1., 2., 3., 4., 5., 6.])  # 1D Array
  t2 = mp.array([[1.,2.,3.], [4.,5.,6.],[7.,8.,9.]]) # 2D Array 
  ```

- 1D & 2D Array with PyTorch

  ``` python
  # 1D Array
  t1 = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
  t1.dim() 	# OUTPUT: rank of t1
  t1.shape 	 
  t1.size()	# OUTPUT: torch.Size(7)
  t1[0] 		# OUTPUT: 해당 인덱스에 위치한 Element 
  t1[2:5]		# OUTPUT: 해당 Index들의 Elements로 이루어진 Tensor
  
  # 2D Array
  t2 = torch.FloatTensor([[0.,1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
  t2.size()	# OUTPUT: torch.Size([행, 열])
  t2[:,1]		# 행,열 순으로 읽으면 됨. 이 경우 모든 행에 있는, 1열의 원소들
  
  ```

- Broadcasting: 

  Matrix는 원칙적으로 크기가 법칙과 안 맞으면 연산(합, 곱)이 불가능하나, 
  다른 크기의 Tensor끼리 연산이 가능하게끔 크기 자동맞춤을 해주는 것이 Broadcasting임

  ``` python
  # Vector(1D) + Scalar(0D)
  m1 = torch.FloatTensor([[1, 2]])
  m2 = torch.FloatTensor([3])
  print(m1 + m2) 
  # m2 Broadcasting : [3] --> [[3, 3]]
  # RESULT :  tensor([4.,5.])
  
  # 2 X 1 Vector + 1 X 2 Vector
  m1 = torch.FloatTensor([[1, 2]])
  m2 = torch.FloatTensor([[3],[4]])
  print(m1 + m2)
  # m1 Broadcasting : [[1, 2]] --> [[1,2],[1,2]]
  # m2 Broadcasting : [[3], [4]] --> [[3, 4], [3, 4]]
  # RESULT : tensor([[4., 5.], [5., 6.]]) 
  # 
  # | 1 |      | 3  4 |  =  | 1  1 |   | 3  4 |   | 4  5 |
  # | 2 |   + 			  | 2  2 | + | 3  4 | = | 5  6 |
  
  ```

  Broadcasting으로 인해 size가 다른 두 행렬을 곱하더라도 프로그램적으로 에러가 나지 않기 때문에, 사용자 입장에선 나중에 잘못된 결과가 나오더라도 알기가 어렵다는 단점이 있다.

  

- Functions

  - 행렬곱 & 산술곱

    ``` python
    m1.matmul(m2) 	# 행렬곱
    m1.mul(m2) 		# 산술곱 / MATLAB으로 치면 .*에 해당하는 것
    ```

    

  - 평균 (Mean)

    ``` py
    t = torch.FloatTensor([1,2])
    print(t.mean())
    # RESULT : tensor(1.5000)
    
    t = torch.FloatTensor([1,2],[3,4])
    t.mean() 		# OUTPUT : tensor(2.5000)
    t.mean(dim= 0) 	# OUTPUT : tensor([2., 3.])
    t.mean(dim= 1)	# OUTPUT : tensor([1.5000, 3.5000])
    t.mean(dim=-1)	# OUTPUT : tensor([1.5000, 3.5000])
     
    ```

    mean(dim=0)의 경우, 열을 기준으로 하여 1과 3의 평균 2, 2와 4의 평균 3이 출력된다.

    아래는 dim이 뜻하는 바에 대한 참고자료.

    ![Meaning_of_Dim](https://i.stack.imgur.com/V3qfN.png)

  - t.sum(), t.max(), t.argmax() 등에 대해서도 마찬가지다. 
    참고로 argmax는 가장 큰 원소의 index를 OUTPUT으로 뱉는다.

- View **매우 중요**

  ``` python
  # numpy에서의 reshape 함수와 동일한 기능이다.
  t = torch.FloatTensor([[[0,1,2],[3,4,5]] , [[6,7,8],[9,10,11]]])
  t.view([-1, 3])		# 
  t.view([-1, 1, 3])	# 
  # 원소의 개수가 view 함수 안에 들어가는 숫자들의 곱으로 나눠지기만 하면 됨.
  ```

  ![](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile22.uf.tistory.com%2Fimage%2F9966623F5B054EBF064C74)

  ![](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile6.uf.tistory.com%2Fimage%2F99E51D3F5B054EC033B6DA)

- Squeeze : view와 유사. 다만 squeeze는 원소가 하나뿐인 dimension을 없앤 Tensor를 뱉어낸다.
  (3X1X4 Tensor라면 3X4로 바꿔준다는 뜻.)

  ``` python
  t.squeeze()
  ```

  

- Unsqueeze : 반대다. 원하는 dimension을 추가할 수 있다. (3X2 Tensor를 3X1X2 Tensor로 변경 가능)

  ``` python
  t = torch.Tensor([0, 1, 2]) # Size: [3]
  t.unsqueeze( 0)				# Size: [1,3]
  t.unsqueeze( 1)				# Size: [3,1]
  t.unsqueeze(-1)				# Size: [3,1]
  # 여기 들어가는 숫자도 위에처럼 그냥 dim 방향을 나타내는 것 같음. 
  # 강의는 뭔소린지 이해가 잘 안 간다...
  ```

  

- Type Casting : 형 변환

  ``` python
  lt = torch.LongTensor([1, 2, 3, 4])
  lt.float()  		# OUTPUT : tensor([1., 2., 3., 4.])
  
  # Byte Tensor : 후에 마스킹 같은 데서 사용함
  bt = torch.ByteTensor([True, False, False, True])
  bt.LongTensor()  	# OUTPUT : tensor([1, 0, 0, 1])
  ```

  

- Concatenate

  ``` python
  x = torch.FloatTensor([[1, 2], [3, 4]])
  y = torch.FloatTensor([[5, 6], [7, 8]])
  
  torch.cat([x,y], dim= 0) # 열 방향으로 이어붙임
  torch.cat([x,y], dim= 1) # 행 방향으로 이어붙임
  ```

  

- Stacking

  ``` python
  x = torch.FloatTensor([1, 4])
  y = torch.FloatTensor([2, 5])
  z = torch.FloatTensor([3, 6])
  
  torch.stack([x,y,z]) # dim=0, 열 방향으로 쌓인 Tensor
  torch.stack([x,y,z], dim= 0) # 행 방향으로 쌓인 Tensor
  # OUTPUT : tensor([[1., 2., 3.],[4., 5., 6.]])
  
  # Same As...
  torch.cat([x.unsqueeze(0), y.unsqueeze(0),z.unsqueeze(0)], dim=0)
  
  ```

  

- Ones and Zeroes

  ``` python
  x = torch.FloatTensor([[0, 1, 2],[2, 1, 0]])
  torch.ones_like(x)		# x와 동일한 모양에 원소가 전부 1인 Tensor 
  torch.zeroes_like(x)	# x와 동일한 모양에 원소가 전부 0인 Tensor
  ```

  

- In-place Operation : 연산하고 바로 밀어넣기

  ``` python
  x.mul(2.)	 # x의 모든 원소에 2 곱해진 Tensor가 OUTPUT임
  x.mul_(2.) 	 # x의 모든 원소에 2 곱해진 Tensor를 x 자리에 바로 넣음
  # Python에선 Memory Allocation이 자동으로 되기 때문에 이런 식으로 메모리 관리를 해주면 
  # 속도 면에서 이점을 볼 수 있다고 생각할 수 있으나
  # PyTorch의 경우 워낙 알아서 잘 관리해주기 때문에 그다지 크게 이득을 보지 못할 수도 있음
  ```

  