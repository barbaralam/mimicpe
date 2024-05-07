import wfdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torch
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

from ecg_classification_utils import BatchDataloader, ResNet1d
from rosemary import metrics_binary_classification

def get_parser():
    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                    description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default='',
                        help='by default consider the ids are just the order')
    parser.add_argument('--age_col', default='age',
                        help='column with the age in csv file.')
    parser.add_argument('--ids_col', default=None,
                        help='column with the ids in csv file.')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                            'The rest is for training')

    return parser


def get_df(data_dir):

    df_neg = pd.read_csv(os.path.join(data_dir, 'EKGneg48.csv'))[:1675]
    df_neg['label'] = 0
    df_pos = pd.read_csv(os.path.join(data_dir, 'EKGpos48.csv'))
    df_pos['label'] = 1
    df = pd.concat([df_neg, df_pos])
    # add path to ecg hea/dat
    ecg_dir = '../data/ecgs/'
    def ecg_path_fn(study_id):
        path = os.path.join(ecg_dir, str(int(study_id)))
        return path
    df['ecg_path'] = df['study_id'].apply(ecg_path_fn)
    df = df[df['ecg_path'].apply(lambda x: os.path.isfile(x+'.hea'))]
    # remove one that cnanot be read by wfdb rdrecord
    df = df[df['study_id'].apply(lambda x: int(x) not in [43010731])]
    # reset index
    df = df.reset_index(drop=True)
    return df


def train(ep, dataload, model, optimizer, device):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for X, y in dataload:
        X, y = X.to(device), y.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y.reshape(y_pred.shape))
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(X)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries




def evaluate(ep, dataload, model, device):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    
    labels = []
    scores = []
    for X, y in dataload:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y.reshape(y_pred.shape))
            #
            labels.append(y)
            scores.append(torch.sigmoid(y_pred.squeeze()))
            # Update outputs
            bs = len(X)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    
    labels = torch.cat(labels)
    scores = torch.cat(scores)
    metrics = metrics_binary_classification(labels, scores)
            
    eval_bar.close()
    eval_loss = total_loss / n_entries
    metrics['eval_loss'] = eval_loss
    
    return metrics


def main(args):
    
    data_dir = '/data/vision/polina/scratch/barbaralam/cxrpe/data/Modeling'
    df = get_df(data_dir)

    paths = df['ecg_path'].tolist()
    ecgs = []
    for path in tqdm(paths):
        try:
            record = wfdb.rdrecord(path) 
            x = record.__dict__['p_signal']
        except:
            print(f'Cannot read {path}')
            x = None
        ecgs.append(x)
    assert(all([x.shape==(5000,12) for x in ecgs]))


    torch.manual_seed(args.seed)
    print(args)
    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    folder = args.folder

    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # Save config file
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')


    X = np.stack([x.T for x in ecgs])
    y = np.array(df['label'].tolist())
    # fill nan with 0
    X = np.nan_to_num(X, nan=0.0)
    # take the middle portion of the signal
    l = (X.shape[2] - args.seq_length)//2
    r = l + args.seq_length
    X = X[:,:,l:r]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('after train_test_split: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    train_loader = BatchDataloader(X_train, y_train, bs=args.batch_size, mask=np.ones(len(X_train)).astype(bool))
    valid_loader = BatchDataloader(X_test, y_test, bs=args.batch_size, mask=np.ones(len(X_test)).astype(bool))
    print('train/val loader len: ', len(train_loader), len(valid_loader))


    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                    blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                    n_classes=N_CLASSES,
                    kernel_size=args.kernel_size,
                    dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                    min_lr=args.lr_factor * args.min_lr,
                                                    factor=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'accuracy', 'auroc', 'precision', 'recall', 
                                ])
    for ep in range(start_epoch, args.epochs):
        train_loss = train(ep, train_loader, model, optimizer, device)
        valid_metrics = evaluate(ep, valid_loader, model, device)
        valid_loss = valid_metrics['eval_loss']
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                    os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break
        # Save history
        row = pd.DataFrame([{"epoch": ep, "train_loss": train_loss,"valid_loss": valid_loss, "lr": learning_rate,
                        'accuracy': valid_metrics['accuracy'], 'auroc': valid_metrics['auroc'],
                        'precision': valid_metrics['precision'], 'recall': valid_metrics['recall'], }])
        history = pd.concat([history, row], ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Print message
        tqdm.write('Epoch {:2d}:  l_train={:.6f}    l_valid={:.6f}    lr={:.7f}\t    acc/auc={:.3f}/{:.3f}'
                .format(ep, train_loss, valid_loss, learning_rate, valid_metrics['accuracy'], valid_metrics['auroc']))
    tqdm.write("Done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)