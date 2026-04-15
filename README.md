Snake Reinforcement Learning (DQN + PyTorch)

Dự án xây dựng game Rắn săn mồi (Snake) với giao diện Pygame và AI học bằng Deep Reinforcement Learning (DQN).

🚀 Giới thiệu

Project này mô phỏng game Snake cổ điển, trong đó:

🧑 Người chơi có thể điều khiển rắn thủ công
🤖 AI có thể tự học cách chơi bằng Deep Q-Network (DQN)
📈 Có biểu đồ theo dõi quá trình học (training)
🎮 Demo
🖥️ Giao diện game

📉 Biểu đồ huấn luyện

⚙️ Cài đặt

Clone repo:

git clone https://github.com/your-username/snake-rl.git
cd snake-rl

Cài thư viện:

pip install -r requirements.txt

Nếu có nhiều Python:

E:/anaconda3/python.exe -m pip install -r requirements.txt
🎯 Chạy game
🧑‍💻 Chơi thủ công
python play.py --mode human --fps 15

Điều khiển:

⬅️: rẽ trái
➡️: rẽ phải
⬆️: đi thẳng
🤖 Chơi bằng AI
python play.py --mode ai --model models/snake_dqn.pth --fps 25
🧠 Huấn luyện model
Train cơ bản
python train.py --episodes 500 --save models/snake_dqn.pth
Kết quả sau khi train
File	Mô tả
snake_dqn.pth	Model tốt nhất
snake_dqn_last.pth	Model cuối
training_plot.png	Biểu đồ điểm
☁️ Huấn luyện trên Kaggle
1. Upload project

Upload:

train.py
snake_rl/
kaggle/train_on_kaggle.py
2. Cài thư viện
pip install pygame numpy torch matplotlib
3. Train
python kaggle/train_on_kaggle.py --episodes 1200
4. Lấy model

Model được lưu tại:

/kaggle/working/models/snake_dqn.pth

Tải về và đặt vào:

models/snake_dqn.pth
⚡ Lệnh nhanh

Train nhanh:

python train.py --episodes 300 --save models/snake_dqn.pth

Train sâu hơn:

python train.py --episodes 1500 --save models/snake_dqn.pth

Chạy AI mượt hơn:

python play.py --mode ai --model models/snake_dqn.pth --fps 30
⚠️ Lỗi thường gặp
❌ Không có pygame
python -m pip install pygame
❌ Không tìm thấy model

Kiểm tra:

models/snake_dqn.pth

👉 Nếu chưa có → cần train trước

🧩 Kiến trúc AI

Model sử dụng:

Deep Q-Network (DQN)
PyTorch
State gồm:
Vị trí thức ăn
Hướng di chuyển
Nguy cơ va chạm
Action:
Rẽ trái
Rẽ phải
Đi thẳng
📈 Ý nghĩa training
Score tăng → AI học tốt
Dao động → đang exploration
Ổn định → policy tốt
🔥 Hướng phát triển
Double DQN
Dueling DQN
Prioritized Replay
Reward shaping
Multi-agent Snake
📌 Gợi ý cấu trúc thư mục
snake-rl/
│
├── models/
├── assets/
│   ├── ui.png
│   └── training_plot.png
├── snake_rl/
├── train.py
├── play.py
└── requirements.txt
👨‍💻 Tác giả

Nguyễn Đình Hà Dương
