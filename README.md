# Snake RL

Game con ran co giao dien Pygame va AI hoc bang Deep Reinforcement Learning (DQN, PyTorch).

## 1. Cai dat moi truong

Mo terminal tai thu muc du an va chay:

```bash
pip install -r requirements.txt
```

Neu may ban co nhieu ban Python, co the dung duong dan python cu the, vi du:

```bash
E:/anaconda3/python.exe -m pip install -r requirements.txt
```

## 2. Chay game ngay

### 2.1 Choi tay de test UI

```bash
python play.py --mode human --fps 15
```

Dieu khien:

1. Mui ten phai: re phai
2. Mui ten trai: re trai
3. Mui ten len: di thang

### 2.2 Chay bang AI

Mac dinh game doc model tai:

1. models/snake_dqn.pth

Lenh chay:

```bash
python play.py --mode ai --model models/snake_dqn.pth --fps 25
```

## 3. Huan luyen model tren may local

Lenh huan luyen co ban:

```bash
python train.py --episodes 500 --save models/snake_dqn.pth
```

Sau khi train xong, thu muc models se co:

1. snake_dqn.pth: model tot nhat
2. snake_dqn_last.pth: model o episode cuoi
3. training_plot.png: bieu do diem theo tung game

## 4. Huan luyen tren Kaggle

### 4.1 Upload project

Upload toan bo source len Kaggle (hoac it nhat cac file sau):

1. train.py
2. thu muc snake_rl
3. kaggle/train_on_kaggle.py

### 4.2 Cai thu vien tren Kaggle

```bash
pip install pygame numpy torch matplotlib
```

### 4.3 Chay train

```bash
python kaggle/train_on_kaggle.py --episodes 1200
```

Model sinh ra tai:

1. /kaggle/working/models/snake_dqn.pth

### 4.4 Tai model ve may

Tai file tren Kaggle va copy vao local:

1. models/snake_dqn.pth

Sau do chay AI local:

```bash
python play.py --mode ai --model models/snake_dqn.pth --fps 25
```

## 5. Lenh mau thuong dung

Train nhanh:

```bash
python train.py --episodes 300 --save models/snake_dqn.pth
```

Train sau hon de AI on dinh hon:

```bash
python train.py --episodes 1500 --save models/snake_dqn.pth
```

Chay AI fps cao hon:

```bash
python play.py --mode ai --model models/snake_dqn.pth --fps 30
```

## 6. Loi thuong gap

1. Bao loi khong tim thay pygame

```bash
python -m pip install pygame
```

2. Bao loi khong tim thay model

Kiem tra file da ton tai chua:

1. models/snake_dqn.pth

Neu chua co, ban can train local hoac train Kaggle truoc.
"# Snake-Refoircement-Learning-" 
