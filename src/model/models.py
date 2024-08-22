import torch
import torch.nn as nn
from torch.nn import functional as F


class DCNN(nn.Module):
    def __init__(self, feature_size, drop, **kwargs):
        super(DCNN, self).__init__(**kwargs)
        # print(feature_size)
        self.dropout = nn.Dropout(drop)
        self.kernel_size = 3
        self.num_steps = 10

        # 输入是(batch_size,feature_size,num_steps)
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=4)
        #
        self.conv2 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=2)
        #
        self.conv3 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=1)
        self.conv4 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=4)
        #
        self.conv5 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=2)
        #
        self.conv6 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=1)
        self.conv7 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=4)
        #
        self.conv8 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=2)
        #
        self.conv9 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                               dilation=1)

        self.conv_1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=1)
        #
        self.conv_2 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=2)
        #
        self.conv_3 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=4)
        self.conv_4 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=1)
        #
        self.conv_5 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=2)
        #
        self.conv_6 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=4)
        self.conv_7 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=1)
        #
        self.conv_8 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=2)
        #
        self.conv_9 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=self.kernel_size,
                                dilation=4)

        self.feature_size = feature_size

        self.lastconv = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=1)
        self.lastconv_ = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=1)

        # self.fc1 = nn.Linear(3*out_c,1)

        # self.fc2 = nn.Sequential(
        #     nn.Linear(feature_size*self.num_steps, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )
        self.fc = nn.Sequential(

            nn.Linear(2 * feature_size * self.num_steps, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # self.fc3 = nn.Sequential(
        #     nn.Linear(feature_size*self.num_steps, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )

    def forward(self, inputs, *args):
        self.len = inputs.shape[0]

        temp = F.pad(inputs, (8, 0))
        temp = self.conv1(temp)
        temp = F.pad(temp, (4, 0))
        temp = self.conv2(temp)
        temp = F.pad(temp, (2, 0))
        outputs = self.conv3(temp)
        outputs = F.relu(outputs)
        inputs2 = outputs + inputs

        temp_ = F.pad(inputs, (2, 0))
        temp_ = self.conv_1(temp_)
        temp_ = F.pad(temp_, (4, 0))
        temp_ = self.conv_2(temp_)
        temp_ = F.pad(temp_, (8, 0))
        outputs = self.conv_3(temp_)
        outputs = F.relu(outputs)
        inputs_2 = outputs + inputs

        temp = F.pad(inputs2, (8, 0))
        temp = self.conv4(temp)
        temp = F.pad(temp, (4, 0))
        temp = self.conv5(temp)
        temp = F.pad(temp, (2, 0))
        outputs = self.conv6(temp)
        outputs = F.relu(outputs)
        inputs3 = outputs + inputs2

        temp_ = F.pad(inputs_2, (8, 0))
        temp_ = self.conv4(temp_)
        temp_ = F.pad(temp_, (4, 0))
        temp_ = self.conv5(temp_)
        temp_ = F.pad(temp_, (2, 0))
        outputs = self.conv6(temp_)
        outputs = F.relu(outputs)
        inputs_3 = outputs + inputs_2

        temp = F.pad(inputs3, (8, 0))
        temp = self.conv7(temp)
        temp = F.pad(temp, (4, 0))
        temp = self.conv8(temp)
        temp = F.pad(temp, (2, 0))
        outputs = self.conv9(temp)
        outputs = F.relu(outputs)
        inputs4 = outputs + inputs3

        temp_ = F.pad(inputs_3, (8, 0))
        temp_ = self.conv7(temp_)
        temp_ = F.pad(temp_, (4, 0))
        temp_ = self.conv8(temp_)
        temp_ = F.pad(temp_, (2, 0))
        outputs = self.conv9(temp_)
        outputs = F.relu(outputs)
        inputs_4 = outputs + inputs_3

        # inputs = inputs[:,:,-1]
        # inputs =inputs.reshape(self.len,feature_size,-1)

        outputs = self.lastconv(inputs4)
        outputs_ = self.lastconv_(inputs_4)

        outputs_total = torch.cat((outputs_, outputs), dim=1)

        outputs_total = outputs_total.reshape(self.len, -1)
        outputs_total = self.fc(self.dropout(outputs_total))
        outputs_total = outputs_total.reshape(len(inputs))

        return outputs_total
