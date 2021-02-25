import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import training_testing_dataset

# load input_data & output_data
training_input_data, testing_input_data, feature = training_testing_dataset.progress_input_data()
training_output_data = training_testing_dataset.deal_with_training_output_data()
testing_output_data = training_testing_dataset.deal_with_testing_output_data()
# print(training_input_data.shape)

# data from numpy change tensor and put into cuda
train_x = torch.from_numpy(training_input_data).type(torch.FloatTensor).cuda()
train_y = torch.from_numpy(training_output_data).type(torch.FloatTensor).cuda()
test_x = torch.from_numpy(testing_input_data).type(torch.FloatTensor).cuda()
EPOCH = 3  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.0001  # learning rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use 'cuda' to calculate

# set training dataset
train_dataset = TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# Build Network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n1_hidden, n2_hidden, n3_hidden, n4_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n1_hidden)
        self.hidden2 = torch.nn.Linear(n1_hidden, n2_hidden)
        self.hidden3 = torch.nn.Linear(n2_hidden, n3_hidden)
        self.hidden4 = torch.nn.Linear(n3_hidden, n4_hidden)
        self.output = torch.nn.Linear(n4_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        x = F.sigmoid(x)
        return x


net = Net(feature, 1000, 1000, 1000, 1000, 1).to(device)  # define the network use cuda

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = torch.nn.BCELoss().cuda()

# training
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.to(device)  # Tensor on GPU
        b_y = y.to(device)  # Tensor on GPU
        # print(x.shape)
        output = net(b_x).view(-1)  # 維度修正
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_output = net(b_x)
        t = Variable(torch.Tensor([0.5])).cuda()  # threshold
        out = (train_output > t).float() * 1
        out_numpy = out.data.cpu().numpy()
        b_y_numpy = b_y.data.cpu().numpy()

        counter = 0
        for i in range(BATCH_SIZE):
            if out_numpy[i] == b_y_numpy[i]:
                counter += 1
        accuracy = counter / BATCH_SIZE * 100
        counter = 0

        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % accuracy)

# # testing(dev.csv)
# test_output = net(test_x)
# t = Variable(torch.Tensor([0.5])).cuda()  # threshold
# out1 = (test_output > t).float() * 1
# out1_numpy = out1.data.cpu().numpy()
#
# counter = 0
# for _ in range(len(out1_numpy)):
#     if out1_numpy[_] == testing_output_data[_]:
#         counter += 1
# accuracy = counter / len(out1_numpy) * 100
# print(f"test accuracy:{accuracy}")


# testing(test.csv)
test_output = net(test_x)
t = Variable(torch.Tensor([0.5])).cuda()  # threshold
out1 = (test_output > t).float() * 1
out1_numpy = out1.data.cpu().numpy()
predict_answer = list(out1_numpy.reshape(-1))


# upload_data
upload_id = training_testing_dataset.id
dict_write={"Id":upload_id, "Category":predict_answer}
mid_term_marks_df = pd.DataFrame(dict_write, columns=["Id", "Category"])
mid_term_marks_df.to_csv("new_sample_submission.csv", index=False)
