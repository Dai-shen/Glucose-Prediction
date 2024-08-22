from numpy import *
from torch.utils.data import DataLoader, Dataset
import torch
import sys


from utils.get_data import get_train_test_data as getdata
from utils.get_dataset import Sugar_dataset
from utils.model_args import get_args

from model.train_test_model import train, test
from model.models import DCNN
from model.model_utils import HuberLoss, init_weights


def main(args):
    data, test_file_list = getdata(args.train_id_list, args.file_path)

    train_dataset = Sugar_dataset(data, args.num_steps, args.feature_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DCNN(feature_size=args.feature_size, drop=args.drop).to(device)
    net.apply(init_weights)

    criterion = HuberLoss(delta=train_dataset.STD())
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    # train
    for i in range(args.epochs):
        train_loss = train(net, train_loader, args.alpha, criterion, optimizer, device)
        print('epoch:', i + 1, '训练各个batch的平均损失:', train_loss / (len(train_dataset) // args.batch_size))
        torch.save(net.state_dict(), args.save_model + f'/{args.model_name}.pth')

    # test
    if test_file_list:   # 用当前训练的模型训练
        for csv_path in test_file_list:
            test_dataset = Sugar_dataset(csv_path, args.num_steps, args.feature_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
            test_loss, pred_sum, actual_sum = test(net, test_loader, criterion, csv_path, args.plot_test, device)
            print(f'测试结果，CNN3在{csv_path}上平均损失为{test_loss / len(test_loader.dataset)}''\n',
                  f'预测高糖次数为{pred_sum}，实际高糖次数为{actual_sum}')


if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
