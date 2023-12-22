import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegressionModel()

# 计算损失+构造优化器
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    # pytorch默认是累加梯度，不是覆盖之前的梯度
    # 所以要先清空，再反向传播
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

print('w', model.linear.weight.item())
print('b', model.linear.bias.item())

x_test = torch.tensor([[5.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)