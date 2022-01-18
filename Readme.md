# Timegan_Tensorflow With GCP

* TimeGan Tensorflow 1.x 모델을 2.x모델로 변경한 것으로 2.x코드에 있는 오류 몇가지를 수정하여 제작하였다.
---
## Requirements required
- Tensorflow 2.3 or Higher
- Numpy 1.19.5
- Pandas
- Python Default Library (datetime, pathlib, pickle, logging, argparser)
---
## Modules
1. gan_train
2. gan_generator
3. settings
4. Modules / dataLoader
5. Modules / location_module
6. Modules / preprocessing
7. Modules / trainer
8. Modules / utils
9. Models / definemodel
---
### 1. gan_train
- 각종 모듈을 한 곳에 모아 프로세스를 완성 시켜 훈련이 되도록 할 수 있는 파일
- 기본적으로 settings에서 설정을 한 후에 gan_train파일을 실행시키면 되는 형태로 되어있음
- Parameters:
	- model_dir: 모델을 저장하는 경로
	- log_dir: 훈련 로그를 저장하는 경로
	- args: 각종 설정에 필요하는 변수들의 들어있는 아규먼트가 들어있음
- Outputs:
	- GAN model
	- model training experiment
	- log
	- Scale pickle


### 2. gan_generator
- train으로 저장된 모델을 이용하여 가짜데이터를 생성하고 저장하는 파일
- 저장은 워크스페이스의 최상위 루트로 저장이 됨

### 3. Settings
- 데이터 경로, 로그 저장 경로, 데이터 크기, 히든레이어 수, 순환신경망 종류, 모델과 스케일  경로 등 설정을 편안하게 할 수있도록 할 수 있는 파일
- GPU  그래픽카드 램 제한 등 저장경로 설정도 해당 부분에 들어가있음
- Args_Prameters:
	- data_path: 데이터가 들어가있는 파일 및 폴더 경로
	- gen_name: 발전기의 명칭을 넣어 해당 발전기만 나오도록 할 수 있음(폴더일 때만 작동/파일로 할 때 오류 걸림)
	- location: 장소마다 전처리 방식이 다름으로 해당 지역 이름을 넣는 곳(westsouth: 서남해, jeju:동복단지)
	- freq: 데이터 리셈플시 시간 단위 설정(T:  분단위, H: 시간 단위)
	- date_start: 날짜를 자를 때 쓰는 시작 지점(Default: None, None시 시작지점 지정 안함)
	- date_end: 날짜를 자를 때 쓰는 끝나는 지점(Defatul: None, None시 시작지점 지정 안함)
	- sel_rnn: 원하는 순환신경망을 지정하는 곳(Default: LSTM, Support: GRU, LSTM)
	- batchnorm: 배치노멀라이저 추가 여부(Default: True, Support: True/False)
	- hidden_dim: LSTM및 GRU의 유닛을 정하는 곳(Default: 64)
	- num_layers: LSTM및 GRU의 층수를 정하는 곳 (Default: 3)
	- seq_len: 시퀀스 길이(시간)을 정하는 곳 (Default: 24)
	- n_seq: 변수의 갯수를 적는 곳(Default: 9 [서남해 기준])
	- batch_size: 배치사이즈 (Default: 128)
	- train_step: 학습 횟수 (Default: 10000)
	- gamma: 판별자 학습 시 지정 숫자 이상 일 때만 로스 값 업데이트(Default: 0.85)
	- lr: Adam Optimizer의 Learning_rate를 지정하는 곳 (Default: 0.001, 훈련셋팅: 0.0002)
	- beta: Adam Optimizer의 beta 개수를 지정 하는 곳 (Default: 0.9, 훈련셋팅: 0.5)
	- beta_2: Adam Optimizer의 beta_2 개수를 지정 하는 곳 (Default: 0.999, 훈련셋팅: 0.999)
	- model_path: gan_train에서 저장한 모델의 경로
	- scale_path: gan_train에서 저장한 스케일의 경로
### 4. Moduels/dataloader
- 데이터로드 역활을 하며, 파일과 폴더 모두 지원한다. 폴더일 경우 폴더 안에 있는 데이터가 같이 곳의 파일이라 가정 혹은 settings에서 gen_name을 지정했을 경우 그 폴더 안에 있는 모든 데이터를 불러와 결합하여 변수에 담는다.

### 5. Moduels/location_module
- 서남해 데이터와 제주에너지공사 동복단지데이터를 기본 전처리 하는 곳이다. freq에 따라 시간단위가 바뀌며 그 사이에 있는 결측 값들은 제거가 된다.

### 6. Moduels/preprocessing
- 사이킷런의 MinMaxSaler와 StandardScaler의 문제점을 개선을 하였다.
- 해당 문제점은 변환 후 역변환할 기존 변환한 shape와 동일하지 않으면 안된다는 문제로 인하여, 카테고리 번호만 알면 해당 카테고리만 역변환 할 수 있도록 제작되었다.
### 7. Moduels/Trainier
- TimeGan에 필요한 훈련 루프가 들어가 있는 파일
### 8. Moduels/utils
- Trainer에 필요한 훈련 코드가 들어가 있는 파일로, 1.x코드를 2.x코드로 변환 시켰다.
- 각 함수 안에 모델이 선언되어있다
### 9. models/definemodel
- 기본적인 모델을 생성하는 파일
- GRU와 LSTM을 지원하며 배치노멀라이저도 settings를 통해 활성화/비활성화 시킬 수 있다
---
## 상세 모델 변경을 원할 때
* 모듈 9번(models/definemodel)에서 DefineModel Class를 수정을 하면된다.

---
# 추가 예정
* 훈련이 끝난 후 검증 그래프 코드 (PCA, T-SNE)