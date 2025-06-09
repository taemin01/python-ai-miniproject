# 미니 프로젝트 보고서

## **환경** 
- Google Colab

## **문제 정의**
- **문제 유형** : 정형 데이터 기반 시계열 회귀 문제
- **목표** : 서울시 기상 데이터의 **과거 7일치 평균 기온** 데이터를 기반으로 **다음날 평균 기온 예측**
- **입력 변수(속성)** : 과거 7일치 평균 기온(temp)
- **출력 변수** : 다음날 평균 기온(temp) 예측값

## **참고 코드**
- 13주차 코드 big14_stock_assign_nofill_2025.ipynb


## **데이터 설명**
- **출처** : [Kaggle] https://www.kaggle.com/datasets/alfredkondoro/seoul-historical-weather-data-2024?select=seoul+2022-01-01+to+2024-01-01.csv  
- **사용 데이터**
    - 서울 기상자료(2018~2024)
    - Alfredo님 제공 데이터 / (9개월 전 업데이트)
- **데이터 개수** : 2018.01.01 ~ 2024.01.01 (총 **2,194개**)
- **사용 속성** : 평균 기온(temp) 1개  
- **데이터 분할 비율** : 전체 80%를 학습 및 검증 데이터로 사용하며, 그 중 20%를 검증 데이터로 분할합니다.  
최종적으로 학습:검증:테스트 = 64:16:20 비율이 되도록 하였습니다.
- **데이터 분할 결과**
  - training : 64% (**1,404개**)
  - val      : 16% (**351개**)
  - test     : 20% (**439개**)
    
## **전처리 과정**
평균 기온 데이터를 0~1 범위로 정규화한 후,  
7일치 데이터를 슬라이딩 윈도우 방식으로 묶어  
과거 7일의 평균 기온을 입력값 X로, 다음날의 평균 기온을 y로  
설정하여 모델 학습용 시계열 데이터를 구성하였습니다.
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# csv 파일 읽어오기
file_name = 'seoul_weather(2018.01.01~2024.01.01).csv'
df = pd.read_csv(file_name, encoding='utf-8')

# 평균 온도 데이터만 추출
temperature = df[['temp']] 

# datetime 컬럼을 문자열에서 datetime 타입으로 변환
df['datetime'] = pd.to_datetime(df['datetime']) 

#평균 온도 데이터를 MinMaxScaler를 사용하여 0~1 범위로 정규화(스케일링)
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(temperature)

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])     # time_steps(7일치) 데이터
        y.append(data[i + time_steps])       # 다음날 데이터
    return np.array(X), np.array(y)

# 시퀀스 길이 설정
time_steps = 7

# train/val/test 데이터를 시퀀스 데이터로 변환
X_train, y_train = create_sequences(train_data, time_steps)
X_val, y_val     = create_sequences(val_data, time_steps)
X_test, y_test   = create_sequences(test_data, time_steps)
```


## **데이터 분할**
```python
total_size = len(scaled)
train_ratio = 0.8
val_ratio = 0.2

train_size = int(total_size * train_ratio)          # 전체 데이터 중 80%
val_size = int(train_size * val_ratio)              # 학습 데이터 중 20%를 검증 데이터로 사용
train_size = train_size - val_size                  # 나머지를 학습 데이터로 사용

train_data = scaled[:train_size]                    # 학습 데이터
val_data = scaled[train_size:train_size + val_size] # 검증 데이터
test_data = scaled[train_size + val_size:]          # 테스트 데이터
```


## **하이퍼 파라미터 최적화 결과**
RNN, LSTM 모델의 하이퍼 파라미터 조합을 테스트한 RMSE 결과입니다.  

일반적으로 LSTM이 장기 시퀀스에 유리하지만, 본 데이터셋은 하루 단위 평균 기온의  
연속성(자기상관성)이 강하지 않거나, 데이터 양이 충분하지 않아 간단한 RNN이 더 좋은 성능을 보입니다.

다른 파라미터 테스트에 비해 비교적 성능 높은 것들로 구성하였습니다.

| RNN | units | activation |  epochs | batch_size | RMSE           |
| :-- |  :---- | :--------- |  :----- | :---------- | :------------- |
| RNN | 64    | tanh        |  100    | 16          | 2.186322476    |
| RNN |  64    | tanh       |  100    | 64          | 2.255072448    |
| RNN |  64    | tanh       |  100    | 32          | 2.340169883    |
| RNN |  64    | tanh       |  200    | 64          | 2.188197357    |
| RNN |  64    | tanh       |  300    | 32          | **2.17677763** |

| LSTM |  units | activation | epochs | batch_size | return_sequences | RMSE        |
| :--- |  :---- | :--------- |  :----- | :---------- | :---------------- | :---------- |
| LSTM |  16    | relu       |  100    | 16          | False                 | 2.901969916 |
| LSTM |  32    | tanh       |  100    | 16          | False                 | 3.121396302 |
| LSTM |  64    | tanh       |  100    | 16          | False                 | 3.050902426 |



## **최종 하이퍼 파라미터**
| RNN | units | activation |   epochs | batch_size | RMSE 
| :-- |  :---- | :--------- |   :----- | :---------- | :------------- | 
| RNN |  64    | tanh       |   300    | 32          | 2.17677763 |


## 그 외 공통 사용 옵션
- **optimizer** : adam
- **learning_rate** : 별도 설정하기 않고 기본값 0.001 사용
- **손실함수** : mean_squared_error(MSE)

## **최종 성능**
- 모델 : RNN (units=64, activation=tanh)
- RMSE : 2.17677763
- **평균적으로 하루 온도 예측 오차가 약 2.18도임을 의미하며, 양호한 성능 수준으로 판단됩니다.**
