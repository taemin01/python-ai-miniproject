# 미니 프로젝트 보고서

### 환경
가상환경 버전 충돌로 인해 Google Colab으로 진행

### 문제 정의
- 정형 데이터 회귀 : 서울시 기상 데이터 평균 온도(temp) 데이터를 이용해  
과거 1주 동안의 평균 온도로 다음 날 평균 온도 예측

- 입력 변수(속성) : 

- 출력 변수 : 

### 데이터 설명
- 출처 : Kaggle   
  https://www.kaggle.com/datasets/alfredkondoro/seoul-historical-weather-data-2024?select=seoul+2022-01-01+to+2024-01-01.csv  

- 사용 데이터 : 서울 기상자료(2018~2024) / Alfredo님의 데이터  

- 데이터 개수 : 2018.01.01~2024.01.01 (총 2194개)
- 사용 속성 : 평균 기온(temp) 1개
- training data, test data 비율 0.2
- training data, valid data 비율 0.2
    
### 전처리 과정
- numpy, pandas 전처리 과정

### 하이퍼 파라미터 최적화 결과
- Test 3개 이상
- 하이퍼 파라미터 예)layer 수, 활성화 함수, 손실 함수, neuron 수, kernel 크기, padding 여부, 배치 사이즈 learning rate 등
- 우선 밑에 표는 바꿀 예정인 값들

| 항목                 | 테스트 값                              |
| :----------------- | :--------------------------------- |
| **batch\_size**    | 16, 32, 64                         |
| **optimizer**      | adam, rmsprop                      |
| **epochs**         | 100, 200                           |
| **learning\_rate** | 기본값, 0.001, 0.0005 (옵션으로 Adam에 적용) |
| **activation**     | relu, tanh (DNN hidden layer 기준)   |

| 항목                 | 적용 값 (테스트 값)               |
| :----------------- | :------------------------- |
| **batch\_size**    | 16, 32, 64                 |
| **optimizer**      | adam, rmsprop              |
| **epochs**         | 100, 200                   |
| **learning\_rate** | 0.001, 0.0005 (adam 내부 설정) |
activation은 DNN에서 relu / LSTM에서 tanh 고정해도 됨

| 실험명  | batch\_size | optimizer | epochs | learning\_rate |
| :--- | :---------- | :-------- | :----- | :------------- |
| 실험 1 | 32          | adam      | 100    | 0.001          |
| 실험 2 | 64          | adam      | 100    | 0.001          |
| 실험 3 | 32          | rmsprop   | 200    | 0.001          |

이런식으로 꾸밀 듯


### 최종 하이퍼 파라미터
- 표 형식으로 꾸미면 될 듯

| 실험명  | batch\_size | optimizer | epochs | learning\_rate |
| :--- | :---------- | :-------- | :----- | :------------- |
| 실험 1 | 32          | adam      | 100    | 0.001          |
| 실험 2 | 64          | adam      | 100    | 0.001          |
| 실험 3 | 32          | rmsprop   | 200    | 0.001          |

최종 파라미터도 이런식으로 작성 예정


### 그 외 옵션
    optimizer : adam  
    손실함수 : mean_squared_error  


### 최종 성능
- 성능 수치 적기
