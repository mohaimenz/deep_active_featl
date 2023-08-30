import os;
import sys;
import numpy as np;
import random;
import time;
from sklearn.utils import shuffle;
from sklearn.model_selection import train_test_split;
import opts as O;

if __name__=='__main__':
    opt = O.parse();
    opt.nClasses=50;
    opt.dataset = 'esc50';
    opt.sr = 20000;
    opt.inputLength = 30225;

    data_dir = os.path.join(opt.data, '{}/data'.format(opt.dataset));
    data = np.load(os.path.join(data_dir, 'deploy/deploy.npz'), allow_pickle=True);
    X = data['x'];
    Y = data['y'];
    deployX, testX, deployY, testY = train_test_split(X, Y, test_size=200, random_state=42);
    np.savez_compressed('{}/deploy/deploy0'.format(data_dir), x=deployX, y=deployY);
    print('Deploy0 data len: {}'.format(len(deployX)));

    #Generate Tune Data
    test_data_dir = os.path.join(data_dir, 'test');
    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir);
    np.savez_compressed('{}/test'.format(test_data_dir), x=testX, y=testY);
    print('Test data len: {}'.format(len(testX)));

    #Generate Tune1=100 and Deploy2=900 samples
    for i in range(1,7):
        deployX, tuneX, deployY, tuneY = train_test_split(deployX, deployY, test_size=100, random_state=42);
        np.savez_compressed('{}/deploy/deploy{}'.format(data_dir, i), x=deployX, y=deployY);
        print('Deploy{} data len: {}'.format(i, len(deployX)));
        np.savez_compressed('{}/tune/tune{}'.format(data_dir, i), x=tuneX, y=tuneY);
        print('Tune{} data len: {}'.format(i, len(tuneX)));

    print('Finished data preparation.');
