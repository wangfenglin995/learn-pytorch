import torch
import matplotlib.pyplot as plt

# 准备数据集
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# 设计模型


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# 构造损失函数和训练器
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_loss = []
epoch_list = []
# 训练循环
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    train_loss.append(loss.item())
    epoch_list.append(epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)

plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(epoch_list, train_loss, linewidth=1, linestyle='solid')
# plt.legend()
plt.show()
