import os
import pandas as pd


def get_train_test_data(train_id, path):
    """
    :param train_id: 训练集的列表，形如 ['02', '04', '06']
    :param path: newdata 相对路径
    :return: 训练集的data, 测试集的若干文件名列表
    """
    train_df = pd.DataFrame()
    all_file_list = os.listdir(path)  # 存储测试集文件列表
    for d in train_id.split(","):
        all_file_list.remove(f"0{d}new.csv")   # 去除拿来训练的数据集
        file_name = f"{path}/0{d}new.csv"
        df = pd.read_csv(file_name)
        train_df = pd.concat([train_df, df], axis=0)

    # 重置行索引
    train_df = train_df.reset_index(drop=True)
    all_file_list = [path + '/' + st for st in all_file_list]
    return train_df, all_file_list


if __name__ == "__main__":
    path = '../../data/newdata'
    train_list = '02,03,04,05,06,07,08,09,10,11,12,13,14,15,16'
    train_data, test_name = get_train_test_data(train_list, path)

