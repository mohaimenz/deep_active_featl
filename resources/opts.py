import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='ACDNet Sound Classification');

    # General settings
    parser.add_argument('--net', default='acdnet',  required=False, choices=['acdnet']);
    parser.add_argument('--dataset', required=False, default='esc50', choices=['esc50', 'us8k', 'iwingbeat']);
    # parser.add_argument('--data', default='{}/datasets/'.format(os.getcwd()),  required=False);
    parser.add_argument('--data', default='/Users/mmoh0027/Desktop/GD-MONASH/phd/experiments/datasets/al/', required=False, help='Path to dataset');
    # parser.add_argument('--data', default='/home/mmoh0027/hy79/md/datasets/al/', required=False, help='Path to dataset');
    # parser.add_argument('--data', default='/home/mmoh0027/mb20/datasets/', required=False, help='Path to dataset');

    #Basic Net Settings
    parser.add_argument('--nClasses', type=int, default=50);
    parser.add_argument('--nFolds', type=int, default=5);
    parser.add_argument('--sr', type=int, default=20000);
    parser.add_argument('--inputLength', type=int, default=30225);

    #Leqarning settings
    parser.add_argument('--batchSize', type=int, default=64);
    parser.add_argument('--weightDecay', type=float, default=5e-4);
    parser.add_argument('--momentum', type=float, default=0.9);
    parser.add_argument('--nEpochs', type=int, default=800);
    parser.add_argument('--LR', type=float, default=0.1);
    parser.add_argument('--warmup', type=int, default=10);

    #Default settings for training pruned models
    parser.add_argument('--retrain', default=False, required=False);

    #Handling unknown arguments
    p, unknown = parser.parse_known_args();
    # print(unknown);
    for i in unknown:
        if i.startswith('--'):
            parser.add_argument(i, default=unknown[unknown.index(i)+1]);

    opt = parser.parse_args();

    opt.retrain = True if opt.retrain == 'True' else False;
    opt.schedule = [0.3, 0.6, 0.9];

    return opt


def display_info(opt):
    print('+------------------------------+');
    print('| {} Sound classification'.format(opt.net.upper()));
    print('+------------------------------+');
    print('| dataset  : {}'.format(opt.dataset));
    print('| nEpochs  : {}'.format(opt.nEpochs));
    print('| LRInit   : {}'.format(opt.LR));
    print('| schedule : {}'.format(opt.schedule));
    print('| warmup   : {}'.format(opt.warmup));
    print('| batchSize: {}'.format(opt.batchSize));
    print('| Classes: {}'.format(opt.nClasses));
    print('+------------------------------+');
