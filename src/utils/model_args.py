import argparse


def get_args():
    parser = argparse.ArgumentParser(description="CNN with Hyperparameters")
    parser.add_argument("--file_path", type=str, required=True,
                        help="the abs_path for train / test data")
    parser.add_argument("--train_id_list", type=str,
                        default='02,03,04,05,06,07,08,09')
    parser.add_argument("--test_id_list", type=str, default='10,11,12,13,14,15,16')

    parser.add_argument("--save_model", type=str, default='/home/daiyf/daiyf/glucose-prediction/src/save_reslut')
    parser.add_argument("--model_name", type=str, default='KCNN')
    parser.add_argument("--load_model", type=str,
                        default='/home/daiyf/daiyf/glucose-prediction/src/oue_result/CNN3.1_New_08_06.pth')

    parser.add_argument("--train_size", type=int, default=2000, help="non-useful")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001, help="for huberloss")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--feature_size", type=int, default=8)
    parser.add_argument("--num_hidden", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--drop", type=float, default=0.)
    parser.add_argument("--alpha", type=int, default=0)
    parser.add_argument("--plot_test", type=bool, default=False)
    return parser.parse_args()
