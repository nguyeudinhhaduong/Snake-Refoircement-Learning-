# 🐍 Snake Reinforcement Learning (DQN + PyTorch)

Dự án xây dựng game **Rắn săn mồi (Snake)** với giao diện **Pygame** và AI học bằng **Deep Reinforcement Learning (DQN)**.

---

## 🚀 Giới thiệu

Project này mô phỏng game Snake cổ điển, trong đó:

- 🧑 Người chơi có thể điều khiển rắn thủ công  
- 🤖 AI có thể tự học cách chơi bằng **Deep Q-Network (DQN)**  
- 📈 Có biểu đồ theo dõi quá trình học (training)

---

## 🎮 Demo

### 🖥️ Giao diện game

![UI](assets/ui.png)

---

### 📉 Biểu đồ huấn luyện

![Training Plot](assets/training_plot.png)

---

## ⚙️ Cài đặt

```bash
git clone https://github.com/your-username/snake-rl.git
cd snake-rl
pip install -r requirements.txt
```

---

## 🎯 Chạy game

### 🧑‍💻 Chơi thủ công

```bash
python play.py --mode human --fps 15
```

### 🤖 Chơi bằng AI

```bash
python play.py --mode ai --model models/snake_dqn.pth --fps 25
```

---

## 🧠 Huấn luyện model

```bash
python train.py --episodes 500 --save models/snake_dqn.pth
```

---

## 📁 Output

- models/snake_dqn.pth  
- models/snake_dqn_last.pth  
- training_plot.png  

---

## ⚠️ Lỗi thường gặp

```bash
python -m pip install pygame
```

---

## 👨‍💻 Tác giả

Nguyễn Đình Hà Dương
