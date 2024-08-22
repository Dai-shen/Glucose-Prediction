import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import statistics
from utils.get_data import get_train_test_data as getdata


class Sugar_dataset(Dataset):
    def __init__(self, data, num_steps, feature_size=8, **kwargs):
        """
        :param data: 训练集为data, 测试集为data_file_name
        :param num_steps: 时间步长
        :param mode:
        :param feature_size:
        :param kwargs:
        """
        super(Sugar_dataset, self).__init__(**kwargs)

        csv = pd.read_csv(data, encoding='utf-8') if type(data) == str else data
        self.csv = csv
        self.num_steps = num_steps
        self.feature_size = feature_size


    def __getitem__(self, idx):  # 这个是处理lastthree的
        # start = idx * self.num_steps    #+1是由于第一行是标题
        # end = (idx+1) * self.num_steps
        # print(self.csv)
        y = torch.tensor(self.csv.iloc[idx, 0])  # +1是由于第一行是标题
        tensor_list = []
        for j in range(1, self.feature_size + 1):
            values_list = np.array(self.csv.iloc[idx, j][1:-1].split(','), dtype=np.float32)  # 去掉中括号
            tensor = torch.tensor(values_list)[np.newaxis, :]
            tensor_list.append(tensor)
        feature = torch.cat(tensor_list, dim=0)
        return feature, y

    def __len__(self):
        num_rows, num_cols = self.csv.shape
        return num_rows  # 这是因为要去掉第一行的特征名称

    def STD(self):
        x = self.csv.iloc[:, 0]
        std = statistics.stdev(x)
        return std

if __name__ == "__main__":
    path = '../../data/newdata'
    train_list = ['02', '04', '05', '06', '07', '09', '10']
    train_data, test_list = getdata(train_list, path)
    dataset = Sugar_dataset(train_data, 10)
    testset = Sugar_dataset(test_list[0], 10)
    print('succ')