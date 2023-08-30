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
    if dataset == 'esc50':
        download_and_create_esc50();
    elif dataset == 'us8k':
        create_us8k();

def download_and_create_esc50():
    """
     Usages: Dataset preparation code for ESC-50
     Prerequisits: FFmpeg and wget needs to be installed.
    """
    mainDir = os.getcwd();
    esc50_path = os.path.join(mainDir, 'datasets/esc50');

    if not os.path.exists(esc50_path):
        os.mkdir(esc50_path)

    sr = 20000;

    # Download ESC-50
    subprocess.call('wget -P {} https://github.com/karoldvl/ESC-50/archive/master.zip'.format(
        esc50_path), shell=True);
    subprocess.call('unzip -d {} {}'.format(
        esc50_path, os.path.join(esc50_path, 'master.zip')), shell=True);
    os.remove(os.path.join(esc50_path, 'master.zip'));

    # Convert sampling rate
    convert_esc50_sr(os.path.join(esc50_path, 'ESC-50-master', 'audio'), os.path.join(esc50_path, 'wav{}'.format(sr // 1000)), sr);

    src_path = os.path.join(esc50_path, 'wav{}'.format(sr // 1000));
    create_esc50_dataset(src_path, os.path.join(esc50_path, 'wav{}.npz'.format(sr // 1000)));

def create_us8k():
    mainDir = os.getcwd();
    dst_path = os.path.join(mainDir, 'datasets/us8k');

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    dst_path = os.path.join(os.getcwd(), 'datasets/us8k')
    fs_list = [20000]

    src_path = os.path.join(mainDir, 'datasets/urbansound8k');
    # Convert sampling rate
    for fs in fs_list:
        convert_us8k_fs(os.path.join(src_path, 'audio'),
                   os.path.join(dst_path, 'wav{}'.format(fs // 1000)),
                   fs)

    # Create npz files
    for fs in fs_list:
        src_path = os.path.join(dst_path, 'wav{}'.format(fs // 1000));
        create_us8k_dataset(src_path, src_path + '.npz')

def convert_esc50_sr(src_path, dst_path, sr):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path);
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path);
        subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_file, sr, dst_file), shell=True);

def convert_us8k_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path);
    for fold in sorted(os.listdir(src_path)):
        if os.path.isdir(os.path.join(src_path, fold)):
            os.mkdir(os.path.join(dst_path, fold))
            for src_file in sorted(glob.glob(os.path.join(src_path, fold, '*.wav'))):
                dst_file = src_file.replace(src_path, dst_path)
                subprocess.call('ffmpeg -i {} -ac 1 -ar {} -acodec pcm_s16le -loglevel error -y {}'.format(
                    src_file, fs, dst_file), shell=True)

def create_esc50_dataset(src_path, esc50_dst_path):
    print('* {} -> {}'.format(src_path, esc50_dst_path));
    esc50_dataset = {};

    for fold in range(1, 6):
        esc50_dataset['fold{}'.format(fold)] = {};
        esc50_sounds = [];
        esc50_labels = [];

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0];
            start = sound.nonzero()[0].min();
            end = sound.nonzero()[0].max();
            sound = sound[start: end + 1]; # Remove silent sections
            label = int(os.path.splitext(wav_file)[0].split('-')[-1]);
            esc50_sounds.append(sound);
            esc50_labels.append(label);

        esc50_dataset['fold{}'.format(fold)]['sounds'] = esc50_sounds;
        esc50_dataset['fold{}'.format(fold)]['labels'] = esc50_labels;

    np.savez(esc50_dst_path, **esc50_dataset);

def create_us8k_dataset(src_path, dst_path):
    print('* {} -> {}'.format(src_path, dst_path))
    dataset = {}
    for fold in range(1, 11):
        dataset['fold{}'.format(fold)] = {};
        sounds = [];
        labels = [];
        for wav_file in sorted(glob.glob(os.path.join(src_path, 'fold{}'.format(fold), '*.wav'))):
            sound = wavio.read(wav_file).data.T[0];
            label = wav_file.split('/')[-1].split('-')[1];
            sounds.append(sound);
            labels.append(label);
        dataset['fold{}'.format(fold)]['sounds'] = sounds
        dataset['fold{}'.format(fold)]['labels'] = labels

    np.savez(dst_path, **dataset)

if __name__=='__main__':
    datasets = ['esc50', 'us8k'];
    for ds in datasets:
        if ds == 'esc50':
            continue;
        print('Preparing {} dataset.'.format(ds));
        # main(ds);
        opt = O.parse();
        opt.nClasses = 50 if ds == 'esc50' else 10;
        opt.nFolds = 5 if ds == 'esc50' else 10;
        opt.dataset = ds;
        opt.sr = 20000;
        opt.inputLength = 30225;

        data_dir = os.path.join(opt.data, '{}/raw-data'.format(opt.dataset));

        if not os.path.exists(data_dir):
            os.mkdir(data_dir);

        mainDir = os.getcwd();
        dataset = np.load(os.path.join(mainDir, 'datasets/{}/wav{}.npz'.format(opt.dataset, opt.sr // 1000)), allow_pickle=True);
        sounds = [];
        labels = [];
        #Generate a balance training set
        for s in range(1, opt.nFolds+1):
            start_time = time.perf_counter();
            sounds.extend(dataset['fold{}'.format(s)].item()['sounds']);
            labels.extend(dataset['fold{}'.format(s)].item()['labels']);

        tvt_indices = [];
        unlbl_pool_indices = [];
        for c in range(opt.nClasses):
            indices = [i for i, v in enumerate(labels) if int(v)==c];
            split_ratio = 0.5 if ds == 'esc50' else 0.115;
            samples_per_class = int(len(indices)*split_ratio);
            tvt_indices.extend(random.sample(indices, samples_per_class));
            unlbl_pool_indices.extend([di for di in indices if di not in tvt_indices]);

        tvtX = [sounds[ti] for ti in tvt_indices];
        tvtY = [labels[ti] for ti in tvt_indices];
        tvtX, tvtY = shuffle(tvtX, tvtY, random_state=42);

        #For large us8k: 50-50 split and then 40% test, 20% val and rest are train
        test_size = 400;
        train_size = 400;
        tvX, testX, tvY, testY = train_test_split(tvtX, tvtY, test_size=test_size, random_state=42);
        trainX, valX, trainY, valY = train_test_split(tvX, tvY, test_size=int(len(tvtX) - (train_size+test_size)), random_state=42);

        np.savez_compressed('{}/train_small'.format(data_dir), x=np.asarray(trainX, dtype=object), y=trainY);
        print('Training data len: {}'.format(len(trainX)));
        np.savez_compressed('{}/val_small'.format(data_dir), x=np.asarray(valX, dtype=object), y=valY);
        print('Val data len: {}'.format(len(valX)));
        np.savez_compressed('{}/test_small'.format(data_dir), x=np.asarray(testX, dtype=object), y=testY);
        print('Test data len: {}'.format(len(testX)));


        #unlblPool1 data having >=50% samples
        unlblPoolX = [sounds[di] for di in unlbl_pool_indices];
        unlblPoolY = [labels[di] for di in unlbl_pool_indices];
        unlblPoolX, unlblPoolY = shuffle(unlblPoolX, unlblPoolY, random_state=42);
        np.savez_compressed('{}/unlblpool_small'.format(data_dir, 0), x=np.asarray(unlblPoolX, dtype=object), y=unlblPoolY);
        print('Unlblpool data len: {}'.format(len(unlblPoolX)));

        print('Finished {} data preparation.'.format(ds));
