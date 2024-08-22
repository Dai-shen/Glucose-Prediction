import torch
from torch.utils.data import DataLoader
from model.model_utils import HuberLoss
from model.models import DCNN
from model.train_test_model import test
from utils.get_dataset import Sugar_dataset
from utils.model_args import get_args

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DCNN(feature_size=args.feature_size, drop=args.drop).to(device)
net.load_state_dict(torch.load(args.load_model))
for csv_path in args.test_id_list.split(","):
    csv_path = args.file_path + '/0' + csv_path + 'new.csv'
    test_dataset = Sugar_dataset(csv_path, args.num_steps, args.feature_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    criterion = HuberLoss(delta=test_dataset.STD())

    test_loss, pred_sum, actual_sum = test(net, test_loader, criterion, csv_path, args.plot_test, device)
    print(f'测试结果，CNN3在{csv_path}上平均损失为{test_loss / len(test_loader.dataset)}''\n',
          f'预测高糖次数为{pred_sum}，实际高糖次数为{actual_sum}')
