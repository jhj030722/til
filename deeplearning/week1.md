## [혼자 공부하는 머신러닝+딥러닝] 18강. 심층 신경망 ▶️인공 신경망에 층을 추가하여 심층 신경망 만들어 보기

![img1](../til/deeplearning/img/img1.png)

- tensorflow가 keras 모델 정의하고 층을 구성성
    - compile 메소드 호출 => 손실함수 등 정의
    - evaluate 구조로 모델 평가 


- 2개의 층
    - 입력층은 입력 데이터 그 자체
    - 2개의 층 : `은닉층`, `출력층` 
    - 28x28 이미지 이므로 784개 뉴런 입력층
    - 다중 분류이기 때문에 softmax 사용, 이진 분류면 sigmoid 
    - 은닉층에 많이 사용되는 `활성화 함수` : sigmoid, relu, 등

```py
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential([dense1, dens2])
```

- param 값이 내가 생각한 구조와 맞아 떨어지면 내가 구성한 대로 잘 동작하는 모델인 것
- output shape 'none' 부분 -> 확률적 경사 하강법 batchsize

#### 층을 추가하는 다른 방법
```py
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
```
(실제로 많이 쓰는 방법)

#### 모델 훈련
```py
model.comfile(loss='sparse_categorical_crossentropy, 
metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
```
- epoch 반복 횟수
