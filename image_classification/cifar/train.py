import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import logging

# 日志记录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_log.txt")

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 训练函数
def train_model(model, train_data, test_data, num_epochs, learning_rate, log_dir, optimizer_type="sgd"):
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)

    # 配置文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, "train_log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    train_losses = []
    test_losses = []

    if optimizer_type == "sgd":
        optimizer = SGD(model.parameters(), learning_rate)
    elif optimizer_type == "adam":
        optimizer = Adam(model.parameters(), learning_rate)
    else:
        raise ValueError("Unknown optimizer type: " + optimizer_type)

    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch_idx in range(0, len(train_data), batch_size):
            data = train_data[batch_idx:batch_idx + batch_size]

            # 前向传播、损失计算和反向传播方法
            predictions = model(data)
            train_loss = cross_entropy_loss(predictions, train_labels)  
            gradients = model.backward(data, train_labels)  # model中要有对应的backward方法

            # 手动更新模型参数
            optimizer.step(gradients)

            total_train_loss += train_loss

        average_train_loss = total_train_loss / len(train_data)
        train_losses.append(average_train_loss)
        logging.info('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, num_epochs, average_train_loss))

        # 测试模型
        test_loss = test_model(model, test_data)
        test_losses.append(test_loss)
        logging.info('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, num_epochs, test_loss))

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(log_dir, f'checkpoint.pkl')
            save_model(model, optimizer, checkpoint_path)

    # 保存训练后的模型参数
    model_path = os.path.join(log_dir, 'model.npy')
    np.save(model_path, model.get_parameters())

    # 绘制损失曲线
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_plot.png'))

# 测试
def test_model(model, test_data):
    total_test_loss = 0
    for data, labels in test_data:
        predictions = model(data)
        test_loss = cross_entropy_loss(predictions, labels)
        total_test_loss += test_loss

    average_test_loss = total_test_loss / len(test_data)
    return average_test_loss


class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self, gradients):
        for param, grad in zip(self.parameters, gradients):
            param -= self.learning_rate * grad


class Adam:
    def __init(self, parameters, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]
        self.t = 0

    def step(self, gradients):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param -= update

# 交叉熵损失函数
def cross_entropy_loss(predictions, labels):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(labels * np.log(predictions))
    return loss



# 示例数据
train_data = np.random.rand(1000, input_dim)  # 替换为你的训练数据
train_labels = np.random.randint(2, size=(1000, output_dim))  # 替换为你的训练标签
test_data = np.random.rand(200, input_dim)  # 替换为你的测试数据
test_labels = np.random.randint(2, size=(200, output_dim))  # 替换为你的测试标签

# 设置超参数
batch_size = 128
epochs = 100
learning_rate = 0.01
input_dim = 784  
output_dim = 10  

# 创建模型实例
model = YourModel(input_dim, output_dim)

# 训练模型（使用SGD优化器）
train_model(model, train_data, test_data, epochs, learning_rate, 'logs', optimizer_type="sgd")
