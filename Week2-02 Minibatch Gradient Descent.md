## Minibatch Gradient Descent

복잡한 머신러닝 모델을 학습하려면 엄청난 양(수십만개)의 데이터를 다룬다.

문제는 이 엄청난 양의 데이터를 한번에 학습시키려면 시간적, 하드웨어적 자원이 부족하다.

따라서 일부만 데이터로 학습하는 방식을 써보겠다.



- 개념

  전체 데이터를 작은 Minibatch 단위로 쪼개어 학습.

  모든 데이터에 대한 Cost를 계산하지 않고 각 minibatch에 있는 데이터만 계산

  

- 효과

  한 번의 업데이트 때마다 계산할 cost 양이 더 적어 업데이트 주기가 빨라짐

  하지만 전체 데이터를 쓰지 않게 때문에 잘못된 방향으로 업데이트를 할 수도 있음(매끄럽지 못함)



- PyTorch Dataset 설정

  ``` python
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
  ```

- PyTorch DataLoader 사용

  ``` python
  from torch.utils.data import DataLoader
  
  dataloader = DataLoader(
  	dataset,		
      batch_size =2,		# Size of each minibatch
      shuffle = True		# 
  )
  ```

  Dataset을 설정한 후에는 이렇게 DataLoader라는 걸 이 데이터셋에 사용할 수 있다.

  - batch_size는 통상 2의 제곱수로 설정하는 편임.
  - 권장: Shuffle = True로 할 시, Epoch마다 데이터셋을 섞어 학습되는 순서를 바꾼다.
    (순서 자체를 학습해버릴 위험이 있음.)



### Full Code with Dataset and DataLoader

``` python
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
    shuffle = True		# 
)


nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader): 
		# enumerate(dataloader)     minibatch 인덱스와 데이터를 받음
        x_train, y_train = samples
        
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Batch{}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
```

