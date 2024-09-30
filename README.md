 # BoostCamp AI Tech Team 12 

# 대회 설명
- 주제 : Sketch 이미지 데이터 분류
Sketch기반 이미지를 분류하여 어떤 객체를 나타내는지 예측하는 대회

<br>

## 팀원 👩🏻‍💻👨🏻‍💻

| 김한별 | 손지형 | 유지환 | 장희진 | 정승민 | 조현준 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/260116cd-e256-412f-a050-c40fe591a114" width="300"> | <img src="https://github.com/user-attachments/assets/6b676bff-a891-48b8-a1f8-2341e9b0b9cf" width="300"> | <img src="https://github.com/user-attachments/assets/7bca579f-f5bd-49be-94be-65dd61f1d71e" width="300"> | <img src="https://github.com/user-attachments/assets/1fc8bf87-5217-457e-b75d-cd9dff1c74ae" width="300"> | <img src="https://github.com/user-attachments/assets/58e0383c-6664-4728-bbbd-2e800b8c6eaf" width="300"> | <img src="https://github.com/user-attachments/assets/c166048d-813f-463d-a590-a47e48ef91ac" width="300"> |
| EDA / Data processing | EDA / Data processing | Modeling / Ensemble | 	Modeling / Ensemble	| Modeling / 코드 모듈화 / wandb 셋팅  / Ensemble | EDA / Data processing  |

** **

## [프로젝트 개요]

Sketch 이미지에 대한 classficiation 대회입니다. 데이터는 총 500개의 클래스로 이루어져있습니다.

해당 대회에서는 사용되는 데이터셋의 원본의 경우 총 50,889개의 데이터셋으로 구성되어있지만, 해당 대회에서는 15,021개의 train 데이터와 10,014개의 private & public 평가 데이터로 나누어 구성되어있습니다.

![sketch](https://github.com/user-attachments/assets/1eaef19e-1e1a-4e9e-b8d5-f60822f132a2)

해당 대회의 특징으로는 실제 이미지가  아닌 스케치 이미지의 분류로, 보다 간결한 선으로 해당 객체를 분류할 수 있어야 한다는 점 입니다.

<br>

## [Classification Model 성능 비교]
![image](https://github.com/user-attachments/assets/216b28e6-6ac2-4d57-bc91-f3a255eaf546)


CoatNet, Efficientnet 위주로 실험을 진행하였습니다. 

** **

## 사용된 기법 

**Best Data Augmentation** : 1. 단순한 Flip 형태의 증강(좌우, 상하, 상하좌우), 2. Rotate 증강(15도 이내) 3. Color Inver ( 색상반전 ) 

**모델** : Coatnet , efficientnetB4, resnet101

**Optimizer** :  Adam | SGD | SAM | AdamW

**Scheduler** : StepLR | OneCycleLR

**생성형 모델** :  Stable Diffusion ([Link](https://github.com/clovaai/CutMix-PyTorch](https://huggingface.co/stabilityai/stable-diffusion-2-1)))

** **

<br>

## 실행 환경
GPU : TESLA V100 <br>
RAM : 32GB <br >
Storage : 100GB <br>
<img src="https://github.com/user-attachments/assets/2d404100-0eeb-41f4-b900-db211183fb22" alt="image" width="70%">

** **

Poetry 설치
<pre><code>curl -sSL https://install.python-poetry.org | python3 -
</code></pre>
<br>
환경 변수 반영
<pre><code>export PATH="$HOME/.local/bin:$PATH"
</code></pre>
<br>
관련 라이브러리 설치
<pre><code>poetry install
</code></pre>
<br>
Poetry 가상환경 실행
<pre><code>poetry shell
</code></pre>

** **

# Train, Test, Ensemble 실행 방법

Train 실행 방법
<pre><code>python train.py --config /path/to/your_train_config.yaml --use_wandb
</code></pre>

<br>

Test 실행 방법
<pre><code>python test.py --config /path/to/your_test_config.yaml
</code></pre>

<br>

Ensemble 실행 방법
<pre><code>python ensemble.py --config /path/to/your/ensemble_config.yaml
</code></pre>


** **
### 실험 및 결과
EfficienNet B4와 CoAtNet3의 val acc에서 90%를 넘는 높은 성능을 보였으며, Test 데이터에 대해서 EfficienNetB4 : 88% | CoAtNet3 : 90.1% 정확도를 보여주었습니다. 이미지 증강의 경우 RGB 색상 Nomalization, Resize(224,224)를 기본 증강으로 사용하였고, Flip, Rotation, Invert, 생성형 이미지를 사용하용하는 것이 가장 높은 성능을 보여주었습니다.
최종적으로 CoAtNet3 + EfficienNetB4 + ResNet + MaxViT ensemble을 통해서 private score : 92.2%를 얻을수 있었습니다.
<div align="center">
  <img src="https://github.com/user-attachments/assets/35999188-1da5-4ff6-b082-39cbcbac1f4d" alt="image1" width="520"/>
 <br>
  <img src="https://github.com/user-attachments/assets/88a68763-64aa-4b1a-ad1a-aad338ccb402" alt="image2" width="520"/>
</div>




** **

## Reference
[1] SAM ([Link](https://github.com/davda54/sam))

[2] clovaai/CutMix-PyTorch ([Link](https://github.com/clovaai/CutMix-PyTorch))
