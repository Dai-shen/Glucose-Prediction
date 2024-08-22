import torch
from matplotlib import pyplot as plt

def plot_test(batch_list, gold_list, pre_list, csv_path):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_list, gold_list, label='gold', color='green')
    plt.plot(batch_list, pre_list, label='prediction', color='red')
    plt.xlabel('batch')
    plt.ylabel('value')
    plt.title(f'CNN3_in_{csv_path}')
    plt.legend()
    plt.grid(True)
    plt.show()


def train(net, train_loader, alpha, criterion, optimizer, device):
    net.train()
    train_loss = 0
    for i, (data, label) in enumerate(train_loader):
        data = data.float()
        data = data.to(device)
        label = label.float()
        label = label.to(device)

        outputs = net(data)
        loss = criterion(outputs, label)
        loss += alpha * torch.clamp(torch.mean(7.8 * 18 - outputs), min=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss


def test(net, test_loader, criterion, csv_path, is_plot, device):
    batch_list, gold_list, pre_list = [], [], []
    test_loss, pred_sum, actual_sum = 0, 0, 0
    net.eval()
    for i, (x, label) in enumerate(test_loader):
        x = x.float()
        x = x.to(device)
        label = label.float()
        label = label.to(device)
        outputs = net(x)

        loss = criterion(outputs, label)
        test_loss += loss.item()
        batch_list.append(i)
        gold_list.append(label.cpu().detach().numpy())
        pre_list.append(outputs.cpu().detach().numpy())

        pred_sum += torch.sum(outputs > 7.8 * 18)
        actual_sum += torch.sum(label > 7.8 * 18)
    if is_plot:
        plot_test(batch_list, gold_list, pre_list, csv_path[-9:-7])
    return test_loss, pred_sum, actual_sum
