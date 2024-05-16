import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from imblearn.over_sampling import SMOTE, ADASYN


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_default_train_val_test_loader(args):

    # get dataset-id
    dsid = args.dataset

    # get dataset from .pt
    data_train = torch.load(f'../../data/Train/{dsid}/X_train.pt')
    data_val = torch.load(f'../../data/Train/{dsid}/X_valid.pt')
    label_train = torch.load(f'../../data/Train/{dsid}/y_train.pt')
    label_val = torch.load(f'../../data/Train/{dsid}/y_valid.pt')
    
    if args.smote:
        data_train = data_train.squeeze().detach().cpu().numpy()
        label_train = label_train.detach().cpu().numpy()
        n_samples, n_features, n_time_steps = data_train.shape
        X_train_reshaped = data_train.reshape(n_samples, n_time_steps * n_features)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, label_train)

        # 如果需要，你可以将过采样后的数据重新转换为3维格式
        data_train = X_train_resampled.reshape(-1, 1, n_features, n_time_steps)
        data_train = torch.from_numpy(data_train).to('cuda:0')
        label_train = torch.from_numpy(y_train_resampled).to('cuda:0')
        print(data_train.shape, label_train.shape)
    elif args.adasyn:
        data_train = data_train.squeeze().detach().cpu().numpy()
        label_train = label_train.detach().cpu().numpy()
        n_samples, n_features, n_time_steps = data_train.shape
        X_train_reshaped = data_train.reshape(n_samples, n_time_steps * n_features)
        
        adasyn = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_reshaped, label_train)
        
        # 如果需要，你可以将过采样后的数据重新转换为3维格式
        data_train = X_train_resampled.reshape(-1, 1, n_features, n_time_steps)
        data_train = torch.from_numpy(data_train).to('cuda:0')
        label_train = torch.from_numpy(y_train_resampled).to('cuda:0')
        print(data_train.shape, label_train.shape)
        

    # init [num_variables, seq_length, num_classes]
    num_nodes = data_val.size(-2)

    seq_length = data_val.size(-1)
    
    num_classes = len(torch.bincount(label_val.type(torch.int64)))


    # convert data & labels to TensorDataset
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_val, label_val)

    # data_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               num_workers=args.workers,)
                                            #    pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.val_batch_size, 
                                             shuffle=False, 
                                             num_workers=args.workers,)
                                            #  pin_memory=True)


    return train_loader, val_loader, num_nodes, seq_length, num_classes
