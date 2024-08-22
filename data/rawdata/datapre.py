import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime


class DataPre:
    # 读取文件，返回迭代器
    def __init__(self, num, load_path, save_path):
        # 特征
        self.num = num
        self.acc = pd.read_csv(load_path + '/' + num + '/ACC_' + num + '.csv', iterator=True, chunksize=10000)
        self.bvp = pd.read_csv(load_path + '/' + num + '/BVP_' + num + '.csv', iterator=True, chunksize=10000)
        self.eda = pd.read_csv(load_path + '/' + num + '/EDA_' + num + '.csv', iterator=True, chunksize=10000)
        self.hr = pd.read_csv(load_path + '/' + num + '/HR_' + num + '.csv', iterator=True, chunksize=10000)
        self.ibi = pd.read_csv(load_path + '/' + num + '/IBI_' + num + '.csv', iterator=True, chunksize=10000)
        self.temp = pd.read_csv(load_path + '/' + num + '/TEMP_' + num + '.csv', iterator=True, chunksize=10000)
        self.food = pd.read_csv(load_path + '/' + num + '/Food_Log_' + num + '.csv', usecols=[2, 9])
        # 金标
        self.dexcom = pd.read_csv(load_path + '/' + num + '/Dexcom_' + num + '.csv', usecols=[1, 7],
                                  skiprows=lambda x: 0 < x < 13)

        # 预处理后的保存路径
        self.save_path = save_path

    # 检查日期是否满足给定的格式（年月日时分秒），删除违规日期
    def is_valid_datetime(self, date_str):
        date_str = str(date_str)
        date_str = date_str.split(".")[0]
        try:
            datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False

    # 将日期转化成时间戳
    def date_to_timestamp(self, date):
        date = str(date)
        date = date.split(".")[0]
        s_t = time.strptime(date, "%Y-%m-%d %H:%M:%S")  # 返回元祖
        mkt = int(time.mktime(s_t))
        return mkt

    # 将数据转换成tensor格式，方便放在GPU上运行，加快速度
    def read_csv_to_tensor(self, df, feature_name):
        time_tensor = torch.tensor(df["datetime"].values, dtype=torch.int64)
        feature_tensor = torch.tensor(df[feature_name].values, dtype=torch.float)
        return time_tensor, feature_tensor

    # 将数据转换成tensor格式，方便放在GPU上运行，加快速度（acc文件）
    def read_csv_to_tensor_acc(self, df):
        time_tensor = torch.tensor(df["datetime"].values, dtype=torch.int64)
        x_tensor = torch.tensor(df[' acc_x'].values, dtype=torch.float)
        y_tensor = torch.tensor(df[' acc_y'].values, dtype=torch.float)
        z_tensor = torch.tensor(df[' acc_z'].values, dtype=torch.float)
        return time_tensor, x_tensor, y_tensor, z_tensor

    # 找到标签前五分钟的特征，每三十秒求一次平均值
    def calculate_feature_average_around_timepoint(self, dexcom_time, time_df, feature_df):
        avg_list = []
        for i in range(-10, 0, 1):
            start_time = dexcom_time + 30 * i
            end_time = start_time + 30
            # 找到第一个子张量中大于1且小于4的元素的索引
            condition = (time_df >= start_time) & (time_df <= end_time)
            # condition = torch.where((df[0] >= start_time) & (df[0] <= end_time), True, False)
            indices = torch.nonzero(condition).squeeze()
            if indices.numel() == 0:
                # 计算第二个子张量中对应索引元素的平均值
                avg_feature = 0.0
            else:
                avg_feature = torch.mean(feature_df[indices].float()).to('cpu').item()
            avg_list.append(avg_feature)
        return avg_list

    # 找到标签前五分钟的特征，每三十秒求一次平均值（food）
    def food_average(self, dexcom_time, df, column):
        avg_list = []
        for i in range(-10, 0, 1):
            start_time = dexcom_time + 30 * i
            end_time = start_time + 30
            if not df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)][column].empty:
                avg_feature = df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)][column].mean()
            else:
                avg_feature = 0.0
            avg_list.append(avg_feature)
        return avg_list

    def forward(self, feature_iter, feature_name):
        time_df = torch.tensor([], dtype=torch.int64).to('cuda:0')
        feature_df = torch.tensor([]).to('cuda:0')
        for chunk in feature_iter:
            chunk = chunk[chunk["datetime"].apply(self.is_valid_datetime)]
            chunk["datetime"] = chunk["datetime"].apply(self.date_to_timestamp)
            chunk0, chunk1 = self.read_csv_to_tensor(chunk, feature_name)
            chunk0 = chunk0.to('cuda:0')
            chunk1 = chunk1.to('cuda:0')
            # chunk = chunk.permute(1,0)
            time_df = torch.cat((time_df, chunk0))
            feature_df = torch.cat((feature_df, chunk1))

        self.dexcom[feature_name + '_average'] = self.dexcom['Timestamp (YYYY-MM-DDThh:mm:ss)'].apply(
            self.calculate_feature_average_around_timepoint, args=(time_df, feature_df,))

        return

    # 将标签对应的时间作为特征返回
    def feature_time(self, dexcom_time):
        s_l = time.localtime(dexcom_time)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", s_l)
        # 将字符串转换为datetime对象
        date_time_obj = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        # 提取时间和分钟
        time_and_minute = date_time_obj.strftime('%H:%M')
        # 去除冒号，得到 "1803"
        time_str_without_colon = time_and_minute.replace(":", "")
        # 将字符串转换为 float 变量
        time_float = float(time_str_without_colon)
        time_list = [time_float for _ in range(10)]
        return time_list

    def preprocess(self):
        # 将标签的日期变成时间戳
        self.dexcom["Timestamp (YYYY-MM-DDThh:mm:ss)"] = self.dexcom["Timestamp (YYYY-MM-DDThh:mm:ss)"].apply(
            self.date_to_timestamp)
        # 处理acc
        time_df = torch.tensor([], dtype=torch.int64).to('cuda:0')
        feature_df = torch.tensor([]).to('cuda:0')
        for chunk in self.acc:
            chunk = chunk[chunk["datetime"].apply(self.is_valid_datetime)]
            # 将时间转化为时间戳
            chunk["datetime"] = chunk["datetime"].apply(self.date_to_timestamp)
            # acc三个方向加权，x，y轴取绝对值，z轴大于等于0乘以2，小于0乘以-0.5，再将三者相加
            time_chunk, x_chunk, y_chunk, z_chunk = self.read_csv_to_tensor_acc(chunk)
            time_chunk = time_chunk.to('cuda:0')
            x_chunk = x_chunk.to('cuda:0')
            y_chunk = y_chunk.to('cuda:0')
            z_chunk = z_chunk.to('cuda:0')
            x_chunk = torch.abs(x_chunk)
            y_chunk = torch.abs(y_chunk)
            z_chunk = torch.where(z_chunk >= 0, z_chunk * 2, z_chunk * (-0.5))
            feature_result = x_chunk + y_chunk + z_chunk
            time_df = torch.cat((time_df, time_chunk))
            feature_df = torch.cat((feature_df, feature_result))

        # 计算每个 Dexcom 时间点前五分钟的 ACC 平均值并添加到 self.dexcom 中
        self.dexcom['acc_average'] = self.dexcom['Timestamp (YYYY-MM-DDThh:mm:ss)'].apply(
            self.calculate_feature_average_around_timepoint, args=(time_df, feature_df,))

        # 计算每个 Dexcom 时间点前五分钟的 BVP, EDA, HR, IBI, TEMP 平均值并添加到 self.dexcom 中
        self.forward(self.bvp, ' bvp')
        self.forward(self.eda, ' eda')
        self.forward(self.hr, ' hr')
        self.forward(self.ibi, ' ibi')
        self.forward(self.temp, ' temp')

        # 处理 food
        df = self.food
        # 处理重复时间戳，求和
        df = df.groupby('time_begin', as_index=False).sum()
        df.set_index('time_begin', inplace=True)
        # 补全时间间隔，生成连续时间序列数据
        freq = '1S'  # 指定时间间隔，例如每小时一个时间戳

        # 重新索引生成连续时间序列数据
        continuous_timestamps = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        food_df = df.reindex(continuous_timestamps, fill_value=0)
        food_df.reset_index(inplace=True)
        new_column_names = {'index': 'datetime'}
        food_df.rename(columns=new_column_names, inplace=True)
        food_df["datetime"] = food_df["datetime"].apply(self.date_to_timestamp)
        df.reset_index(inplace=True)
        df["time_begin"] = df["time_begin"].apply(self.date_to_timestamp)

        # 计算每个时间点对应的碳水化合物含量
        for index, row in df.iterrows():
            # intake_time:摄入时间 begin_time:开始消化的时间 end_time:消化结束的时间
            intake_time = row['time_begin']
            total_carb = int(row['total_carb'])
            begin_time = intake_time + 15 * 60
            end_time = begin_time + total_carb * 120

            intake_mask = (food_df['datetime'] >= begin_time) & (food_df['datetime'] <= end_time)
            food_df.loc[intake_mask, 'total_carb'] += total_carb

            consume_mask = (food_df['datetime'] > begin_time) & (food_df['datetime'] <= end_time)
            food_df.loc[consume_mask, 'total_carb'] -= 0.5 / 60 * np.arange(1, sum(consume_mask) + 1)

        self.dexcom['carb_average'] = self.dexcom['Timestamp (YYYY-MM-DDThh:mm:ss)'].apply(
            self.food_average, args=(food_df, 'total_carb',))

        # 时间
        self.dexcom['feature_time'] = self.dexcom['Timestamp (YYYY-MM-DDThh:mm:ss)'].apply(
            self.feature_time)

        # 删除特征时间与金标时间无法对齐的行
        for i in range(2, 8):
            self.dexcom = self.dexcom[
                ~self.dexcom.iloc[:, i].apply(lambda x: x == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

        # 重新设置索引
        self.dexcom.reset_index(drop=True, inplace=True)
        # 去掉时间列
        self.dexcom = self.dexcom.drop(['Timestamp (YYYY-MM-DDThh:mm:ss)'], axis=1)
        self.dexcom.to_csv(self.save_path+'/'+self.num+'new.csv', index=False)

        return
