import argparse


def get_args():
    parser = argparse.ArgumentParser(description="for data preprocessing")
    parser.add_argument("--load_path", type=str, required=True, help="rawdata")
    parser.add_argument("--save_path", type=str, required=True, help="newdata")
    parser.add_argument("--person_list", type=str, default='004,005,006,007')
    return parser.parse_args()
