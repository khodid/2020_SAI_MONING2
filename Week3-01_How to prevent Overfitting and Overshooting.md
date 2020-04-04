# 이론



## Maximum Likelihood Estimation(MLE)

한국 말로는 '최대 가능도 추정'이라고 부른다.

Likelihood란 무엇이고, MLE를 왜 하는 것일까?

 압정을 땅에 던졌을 때 압정이 떨어진 모양을 두 가지 케이스로 나누는 걸 생각해보자.  납작한 부분이 바닥에 완전히 닿은 경우 Class 1, 뾰족한 부분이 바닥을 짚는 경우를 Class 2라고 생각해보자. 물론 물리적으로 정확한 확률 분포가 있기야 할 것이다. 하지만 우리는 이걸 머신러닝을 통해 구해보도록 하겠다.

 Binomial distribution이기 때문에 압정이 떨어진 모양은 베르누이 분포를 가질 것이다. 이 베르누이 분포를 갖는 동작을 여러 번 반복하면 이항분포로서 나타난다. (아래 수식)
$$
K \sim \mathcal B (n, \theta)P(K = k) = {n \choose k} \theta^k(1-\theta)^{n-k}= \frac{n!}{k!(n-k)!}\cdot \theta^k (1-\theta)^{n-k}
$$
n과 k는 우리의 관찰 결과로 채워 넣으면, 이 함수는 theta에 따른 함수인데, 이걸 그래프로 그리면 theta에 따른 likelihood를 나타내는 언덕 모양이 된다. 우리는 이 likelihood가 가장 큰 지점의 theta를 찾아야 한다. 그래서 경사하강법과 매우 유사한 방식으로 최대점을 찾는 것이다.(Optimization via Gradient Descent) 대충 이런 모양이 될 거다.
$$
\theta := \theta - \alpha \nabla_\theta L(x;\theta)
$$
뭐 이런 식으로 θ를 업데이트한다.



만약 연속적이고 가우시안 분포(=정규 분포)였다면 f(θ)  가 아니라 f(μ,σ)로 구해야 할 것이다.

음......... 추후에 보충해야 할 것 같다. [일단 참고할 만한 링크 저장](https://ratsgo.github.io/statistics/2017/09/23/MLE/)



### MLE의 Overfitting 문제

 MLE를 쓸 경우, 가장 확률적으로 적합한 모델을 기계적으로 찾는 방법이기 때문에  조금 불균질한 튀는 데이터에 대해서도 전부 수용할 수 있는 복잡한 모델을 내놓을 수가 있다. 이렇게 학습 데이터에 과도하게 딱 맞는 모델이 나오는 상황을 *Overfitting* 이라고 부른다.

 이를 해결하는 방법에 대해선 [밑](##Overfitting-방지)에서 설명하겠다.





# 머신러닝 모델 설계 Tips

## Learning rate 조절

Cost Function을 최소화하기 위한 Gradient Descent Algorithm을  복기해보자.


$$
-\alpha \vartriangle \mathcal L (w_1, w_2)
$$
에서 α는 프로그램에서 우리가 lr(learning rate)으로 지정해주었던 . 이 α 값을 어느 정도로 정해줘야 가장 괜찮은 모델을 뽑을 수 있을까?

- α가 너무 **작다면** : 학습이 너무 더뎌서 많은 epoch가 필요함

- α가 너무 **크다면** : 최솟값으로 다가가질 않고 건너뛰는, **overshooting** 발생 가능성 있음. (따라서 혹시 cost가 작아지지 않고 계속 발산하기만 한다면 learning rate을 너무 크게 잡진 않았는지 검토해봐야 함.)  

  ![경사하강법](https://jaehyeongan.github.io/image/learning_rate_sl.png)



학습률 조절에 관한 시각적인 실습을 [여기서](https://developers.google.com/machine-learning/crash-course/fitter/graph?hl=ko) 간단하게 해볼 수 있다.

검색을 해보니, 차라리 overshooting은 바로 알아차릴 수나 있어서 learning rate이 너무 작은 경우에 비해 더 괜찮은 문제라고 한다. 그리고 적절한 learning rate을 찾는 일반론적인 방법은 딱히 없다고 한다. 여러 번 해보는 게 최선이라고.



## Data Preprocessing (데이터 선처리)

데이터 선처리를 해야 할 경우가 있다.

- Data preprocessing for gradient descent:
  y = w1*x1 + w2*x2 + b 와 같은 모델을 이용해 학습하는 경우를 살펴 보자.
  w1, w2를 변수로 갖는 장은 대략 한 곳을 최저점으로 삼는 움푹 패인 모양이 될 것임.

  ![got this image from youtube comments](https://cdn-images-1.medium.com/max/1600/1*mc-P5QwXIVpUJpnHtmwSBg@2x.png)

  ![Data preprocessing for gradient descent](https://t1.daumcdn.net/cfile/tistory/2665A54957996A9031)

  이런 모양을 갖는 것이 이상적이다.
  그러나 실제로는 x1과 x2의 분산 값이 서로 비슷하지 않다면, 저게 균일한 원형이 되지 않을 것이다.
  ![Data preprocessing for gradient descent](https://t1.daumcdn.net/cfile/tistory/2564944957996A9132)

   이런 모양이 될 경우 한 축에 대해서 α 값을 잘 조정해 주었더라도 다른 축에 대해선 너무 큰 값이 되어 위 그림에서 w2축에 대해 overshooting(?)이 발생한 것처럼  cost가 수렴하지 않고 튕겨다니는 걸 볼 수 있다.
   이를 해결하기 위해 normalize 를 한다. 다음에 소개하는 것은 그 방법 중 하나다.

  - Standarization

     수식으로는, 
    $$
    x'_j = \frac{x_j - \mu_j}{\sigma_j}
    $$
     이렇게 표현되며, 실제 코드로 옮길 땐

    ```python
    x_std[:,0] = (x[:,0] - x[:,0].mean) / x[:,0].std() 
    ```

    이런 식이다.

    어디서 많이 봤다 했더니 고등학교 수학에서 정규분포를 표준화하는 공식과 동일한 것 같다.





## Overfitting 방지

머신러닝의 가장 큰 골칫거리라고 한다.

'학습 데이터' 그 자체에 너무 딱 맞는 모델로 발전했을 경우를 일컫는다.

학습 데이터로 실험을 해보면 cost도 적고 잘 맞는 것처럼 보이는데, 실전에 투입해서 다른 데이터를 넣어보면 결과가 썩 만족스럽지 않은 현상이 나타난다.

![Stackoverflow 답변 중에서 찾음](https://i.stack.imgur.com/ZTQSP.png)

위 그림에서 두 Error(Loss) 간의 거리가 멀어지는 지점을 overfitting 되는 지점으로 생각한다. 성공적인 모델을 만들기 위해선 위 그림에서 High Bias와 High Variance 사이의 구간에 해당하는 모델을 선택해야 한다.



- 해결방법

  - 트레이닝 데이터를 굉~장히 많이 쓴다
    (적은 데이터 셋 환경에서 소수의 튀는 값들이 더 큰 영향을 주기 때문. 표본이 많을 수록 모집단과 비슷하다.) 

  - feature(데이터를 설명하는 특징)의 개수를 될 수 있는 한 줄여준다. 

  - **Regularization(일반화)** - 여러 방법이 있다.

    - *Early stopping* : 모델이 너무 복잡해지기 전에, Validation Loss가 더이상 낮아지지 않는 지점에서 훈련을 중단하는 방법 

    -  *L2 Regularization* : weight에 너무 큰 수를 넣지 않는 방법. 너무 구불구불하고 자세한 모델을 만들지 않도록...
      Cost 함수를 이렇게 만드는 식이다.
      $$
      \mathcal L = \frac{1}{N} \sum_i \mathcal D \left( S(Wx_i +b) \right)+ \lambda \sum W^2
      $$
      즉 각각의 element에 대한 W가 클 수록 Cost가 증가하도록 하는 것.

      이 때 λ 를 *regularization strength*라고 부르고, 모델을 단순하게 만드는 걸 얼마나 중요하게 생각하느냐를 반영할 수 있는 상수다.

       [그렇다면 이 람다 값은 어떻게 정할까?](https://stackoverflow.com/questions/12182063/how-to-calculate-the-regularization-parameter-in-linear-regression) 

      위 링크의 Stack Overflow 답변에 의하면, 이 역시 Training data의 일부를 λ=0에서부터 값을 점점 키우면서 돌려 보면서 모델이 예측값을 어떻게 내놓는지를 관찰()하며 결정해야 한다. 그리고 결정한 값에서 살짝 작게 잡아야 전체 데이터에 맞을 것...이라고 말하는 것 같다.

      더 나아가면, 람다 값을 직접 관찰하며 임의로 정해 줘야 한다는 모호함을 피하고 싶다면 Tikhonov Regularization 이라는 다른 일반화 방법은 써야 하는 상수값에 대한 솔루션을 정해줄 수 있으니 그 쪽을 권한다는 답변도 있다. 

    - 딥러닝에선 Neural Network 크기를 줄이는 방법이 유효하다.

    - 가장 많이 사용되는 Dropout, Batch normalization 이라는 방법이 있는데, 이 역시 딥러닝에서 사용되는 기술이다. 추후에 다루겠다고 함.



### DNN에서의 오버피팅 방지

- Deap Neural Network의 과정

  1. 입력 데이터가 1D vector고 10개의 feature가 있다고 하면, 5개 유닛을 가진 softmax layer가 나올 것이다. 

  2. 오버피팅이 될 때까지 size를 늘려 나간다. (input, output은 고정한 채로 중간의 깊이와 너비만 확장) : Training Set의 Loss는 낮아지면서, Validation Set의 Loss는 높아지기 시작할 때가 overfitting 될 때이니 이걸 확인하고,  regularization 방법(drop-out, batch-normalization)을 추가해준다.
  3. 2의 과정을 반복한다.





## 머신러닝 모델을 평가하는 방법

학습을 충분히 시킨 모델이 얼마나 적합한지 평가하려면 어떻게 해야 할까?

학습 데이터 세트를 그래도 평가에 갖다 쓰는 건 유효하지 않다.

따라서 모델이 한 번도 접해보지 않은, Test set이라는 새로운 데이터 뭉치가 필요하다.

실전에선 training set과 testing set을 어느 정도 비율로 섞어서 모델을 검증하는 편이라고 한다.

- 정확도(Accuracy) 평가
  - 단순하게도, 예측으로 정답을 얼마나 맞췄는지를 비율로 나타낸다.
  - 이미지 인식 분야의 경우 95~99% 정도의 정확도를 가짐.



### Training / Testing Data Set

- Training Set (비율 0.8)

  - Training set : 모델을 학습시킴
  - Validation set (비율 0~0.1) : Development set이라고도 부른다. 
    기능 (1) α, λ과 같은 조절하는 역할의 상수들을 모의 시험을 해보며 어떤 값이 제일 좋을지 튜닝함. 
    기능 (2) 우리는 궁극적으로는 test set을 잘 맞추는 모델을 설계하길 목표로 하기 때문에, test set과 얼마나 맞는지 따지며 Training하기를 '반복'한다면 Test Set에 대해 과적합이 될 가능성도 있다. 그래서 Training Set에 대해 development set으로 새 데이터에 대한 적용 가능성을 테스트하고, 그 다음에 test set을 접하게 만들면 훨씬 정확한 모델을 만들 수 있다. 하지만 이쪽에 데이터 분량을 얼마나 할당해 줄지는 상황에 따라 다르다!

- Test Set (비율 0.1~0.2)

  학습 단계에선 절대 개입하지 않는, 정확도 평가용 데이터 셋.



### Online Learning

방대한 분량의 학습 데이터 셋을 여러 묶음으로 쪼개서 (A, B, C, D, ... 라고 부르겠다.)

A set에 대해 학습을 시키고, 그 모델을 이어받아 B set에 대해 학습을 시키고, 이전 결과를 또 이어받은 상태로 C set도 학습시키고... 하는 방식을 *Online Learning*이라고 부른다.

이전 데이터를 그대로 답습하는 게 아니라 추가로 학습을 시키는 점에서 장점이 있다.



# 실습

여기다가도 따로 정리하고 싶었는데 시간적/에너지적 여유가 없어서 collab으로 정리한 걸 그대로 가져오도록 하겠음.



[Lab 07-1 Overfitting & Overshooting](https://github.com/khodid/2020_SAI_MONING2/blob/master/lab07-1_code_Overfitting_and_OverShooting.ipynb)

- 내용 : 
  1. Training data에 Overfitting되는 걸 확인하는 방법 알아보기
  2. Learning rate을 너무 낮게 / 높게 설정했을 때의 cost값 변화 관찰하기
  3. 데이터 전처리 방법을 배우고, 전처리를 하지 않았을 때의 결과와 비교하기



[Lab 07-2 How to use MNIST dataset](https://github.com/khodid/2020_SAI_MONING2/blob/master/lab07-2_code_Using_MNIST_dataset.ipynb)

- 내용 :
  1. MNIST dataset을 import하고 PyTorch에 맞게 가공하는 방법 알아보기
  2. 굉장히 많은 수의 데이터를 불러와서 학습시키고 테스트하는 프로그램의 스크립트 구조 배우기

