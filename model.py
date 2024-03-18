from torch import nn


class Net(nn.Module):
    def __init__(self,net_k_inp,net_k_out):
        #1680 151200
        self.input_size = net_k_inp
        self.hidden_size1 = 1000
        self.hidden_size2 = 1000
        # Количество узлов на скрытом слое
        self.num_classes = net_k_out  # Число классов на выходе. В этом случае от 0 до 9
        self.num_epochs = 10  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        self.learning_rate = 0.01  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net, self).__init__()  # Наследуемый родительским классом nn.Module
        # self.layer1 = nn.Sequential(nn.Conv2d(1, 45, kernel_size=5, stride=1, padding=2),
        #                             nn.ReLU(), nn.MaxPool2d(kernel_size=10, stride=4))
        # self.layer2 = nn.Sequential(nn.Conv2d(45, 45 * 2, kernel_size=5, stride=1, padding=2),
        #                             nn.ReLU(), nn.MaxPool2d(kernel_size=10, stride=4))
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.input_size,
                             self.hidden_size1)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
        self.fc2 = nn.Linear(self.hidden_size1,
                             self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2,
                             self.num_classes)

        # -----------------------------------------------------------------------------------

    def forward(self, x):  # Передний пропуск: складывание каждого слоя вместе
        # out = self.layer1(x)
        # out = self.layer2(out)
        # # out = out.reshape(out.size(0), -1)
        # out = out.view(-1)
        # out = self.drop_out(out)
        out=x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        # out = self.relu(out)
        # out = out.reshape(out.size(1), -1)
        return out


class Net3(nn.Module):
    def __init__(self):
        self.input_size = 36720
        self.hidden_size1 = 1000
        # Количество узлов на скрытом слое
        self.num_classes = 5  # Число классов на выходе. В этом случае от 0 до 9
        self.num_epochs = 1  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        self.learning_rate = 0.0001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net3, self).__init__()  # Наследуемый родительским классом nn.Module
        self.layer1 = nn.Sequential(nn.Conv2d(1, 45, kernel_size=5, stride=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=10, stride=5))
        self.layer2 = nn.Sequential(nn.Conv2d(45, 45*2, kernel_size=5, stride=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=5, stride=3))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.input_size,
                             self.hidden_size1)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
        self.fc2 = nn.Linear(self.hidden_size1,
                             self.num_classes)

        # -----------------------------------------------------------------------------------

    def forward(self, x):  # Передний пропуск: складывание каждого слоя вместе
        out = self.layer1(x)
        out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        out = out.view(-1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.relu(out)
        # out = out.reshape(out.size(1), -1)
        return out


class Net2(nn.Module):
    def __init__(self):
        self.input_size = 149940
        self.hidden_size1 = 1000
        self.hidden_size2 = 1000 * 3
        self.hidden_size3 = 1000 * 3
        self.hidden_size4 = 1000
        # Количество узлов на скрытом слое
        self.num_classes = 5  # Число классов на выходе. В этом случае от 0 до 9
        self.num_epochs = 1  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        self.learning_rate = 0.0001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net2, self).__init__()  # Наследуемый родительским классом nn.Module
        self.layer1 = nn.Sequential(nn.Conv2d(3, 45, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=6, stride=3))
        self.layer2 = nn.Sequential(nn.Conv2d(45, 45 * 2, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=6, stride=3))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.input_size,
                             self.hidden_size1)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
        self.fc2 = nn.Linear(self.hidden_size1,
                             self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2,
                             self.hidden_size3)
        self.fc4 = nn.Linear(self.hidden_size3,
                             self.hidden_size4)
        self.fc5 = nn.Linear(self.hidden_size4,
                             self.num_classes)

        # -----------------------------------------------------------------------------------

    def forward(self, x):  # Передний пропуск: складывание каждого слоя вместе
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        # out = out.reshape(out.size(1), -1)
        return out
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.num_epochs = 1
#         self.learning_rate=0.01
#         self.l1 = nn.Linear(1000, 500)
#         self.l2 = nn.Linear(500, 100)
#         self.l3 = nn.Linear(100, 32)
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.l1(x)
#         x = self.relu(x)
#         x = self.l2(x)
#         x = self.relu(x)
#         x = self.l3(x)
#         x = self.sig(x)
#         return x