import os;
import sys;
import numpy as np;
import random;
import time;
import subprocess;
import glob;
import wavio;
from sklearn.utils import shuffle;
from sklearn.model_selection import train_test_split;
import opts as O;

def main(dataset):
    create_insectwingbeat();

def create_insectwingbeat():
    mainDir = os.getcwd();
    sr = 20000;

    ds_dir = os.path.abspath('..');
    src_path = os.path.join(ds_dir, 'datasets/iwingbeat');
    # # Convert sampling rate
    # convert_iwingbeat_sr(src_path+'/original_audio', os.path.join(src_path, 'wav{}'.format(sr // 1000)), sr);

    # Create npz files
    src_path = os.path.join(src_path, 'wav{}'.format(sr // 1000));
    convert_iwingbeat_dataset(src_path, src_path + '.npz');


def convert_iwingbeat_sr(src_path, dst_path, sr):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    print('* {} -> {}'.format(src_path, dst_path));
    print(sorted(os.listdir(src_path)));
    for fold in sorted(os.listdir(src_path)):
        if os.path.isdir(os.path.join(src_path, fold)):
            os.mkdir(os.path.join(dst_path, fold))
            for src_file in sorted(glob.glob(os.path.join(src_path, fold, '*.wav'))):
                dst_file = src_file.replace(src_path, dst_path)
                subprocess.call('ffmpeg -i {} -ac 1 -ar {} -acodec pcm_s16le -loglevel error -y {}'.format(
                    src_file, sr, dst_file), shell=True)


def convert_iwingbeat_dataset(src_path, dst_path):
    print('* {} -> {}'.format(src_path, dst_path))
    dataset = {}
    folds = sorted(os.listdir(src_path));
    if '.DS_Store' in folds:
        del folds[0];
    print(folds);
    for fold in folds:
        dataset['{}'.format(fold)] = {};
        sounds = [];
        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}'.format(fold), '*.wav'))):
            sound = wavio.read(wav_file).data.T[0];
            sounds.append(sound);
        dataset['{}'.format(fold)]['sounds'] = sounds

    np.savez(dst_path, **dataset)

if __name__=='__main__':
    ds = 'iwingbeat';
    print('Preparing {} dataset.'.format(ds));
    # main(ds);
    opt = O.parse();
    opt.dataset = ds;
    opt.nClasses = 10;
    opt.dataset = ds;
    opt.sr = 20000;
    opt.inputLength = 20000; # each audio is 1 sec

    data_dir = os.path.join(opt.data, '{}/raw-data'.format(opt.dataset));
    print(data_dir);
    if not os.path.exists(data_dir):
        os.mkdir(data_dir);

    mainDir = os.path.abspath('..');
    dataset = np.load(os.path.join(mainDir, 'datasets/{}/wav{}.npz'.format(opt.dataset, opt.sr // 1000)), allow_pickle=True);
    sounds = [];
    labels = [];
    #Generate a balance training set
    key_list = list(dataset.keys());
    for key in key_list:
        x = dataset[key].item()['sounds'];
        y = [key_list.index(key)] * len(x);
        sounds.extend(x);
        labels.extend(y);

    tvt_indices = [];
    unlbl_pool_indices = [];
    for c in range(opt.nClasses):
        indices = [i for i, v in enumerate(labels) if int(v)==c];
        split_ratio = 0.35;
        samples_per_class = int(len(indices)*split_ratio);
        tvt_indices.extend(random.sample(indices, samples_per_class));
        unlbl_pool_indices.extend([di for di in indices if di not in tvt_indices]);

    tvtX = [sounds[ti] for ti in tvt_indices];
    tvtY = [labels[ti] for ti in tvt_indices];
    tvtX, tvtY = shuffle(tvtX, tvtY, random_state=42);

    #For insect wing beat dataset: 35-65 split and then 5000 test, 2500 val and rest are train
    test_size = 5000;
    train_size = 10000;
    tvX, testX, tvY, testY = train_test_split(tvtX, tvtY, test_size=test_size, random_state=42);
    trainX, valX, trainY, valY = train_test_split(tvX, tvY, test_size=len(tvX) - train_size, random_state=42);

    np.savez_compressed('{}/train'.format(data_dir), x=np.asarray(trainX, dtype=object), y=trainY);
    print('Training data len: {}'.format(len(trainX)));
    np.savez_compressed('{}/val'.format(data_dir), x=np.asarray(valX, dtype=object), y=valY);
    print('Val data len: {}'.format(len(valX)));
    np.savez_compressed('{}/test'.format(data_dir), x=np.asarray(testX, dtype=object), y=testY);
    print('Test data len: {}'.format(len(testX)));


    #unlblPool1 data having >=65% samples
    unlblPoolX = [sounds[di] for di in unlbl_pool_indices];
    unlblPoolY = [labels[di] for di in unlbl_pool_indices];
    unlblPoolX, unlblPoolY = shuffle(unlblPoolX, unlblPoolY, random_state=42);
    np.savez_compressed('{}/unlblpool'.format(data_dir, 0), x=np.asarray(unlblPoolX, dtype=object), y=unlblPoolY);
    print('Unlblpool data len: {}'.format(len(unlblPoolX)));

    print('Finished {} data preparation.'.format(ds));
