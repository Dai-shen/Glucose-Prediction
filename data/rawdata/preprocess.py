import datetime

from datapre import DataPre
from src.utils.data_args import get_args


def main(args):
    nums = args.person_list.split(',')
    for num in nums:
        print(f"{str(datetime.datetime.now()).split('.')[0]} 开始数据预处理：{num}")
        person = DataPre(num=num, load_path=args.load_path, save_path=args.save_path)
        person.preprocess()
        print(f"{str(datetime.datetime.now()).split('.')[0]} {num} 已完成!")


if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
