import os;
import sys;
import numpy as np;
import random;
import time;
from sklearn.utils import shuffle;
from sklearn.model_selection import train_test_split;

import utils as U;
import opts as O;

class Generator():
    #Generates data for Keras
    def __init__(self, samples, labels, options, train=True):
        random.seed(42);
        #Initialization
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))];
        self.opt = options;
        self.batch_size = self.opt.batchSize;
        self.train = train;
        self.preprocess_funcs = self.preprocess_setup();
        self.data_indices = list(range(len(self.data)));

    def reset_indices(self):
        self.data_indices = list(range(len(self.data)));

    def get_batch(self, idx):
        #Generate one batch of data
        batchX, batchY = self.generate_batch(idx);
        batchX = np.expand_dims(batchX, axis=1);
        batchX = np.expand_dims(batchX, axis=1);
        return batchX, batchY

    def generate_batch(self, batchIndex):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        indexes = None;
        for i in range(self.batch_size):
            if self.train:
                #Make sure all the samples are exposed to every epoch
                sound1_idx = random.sample(self.data_indices, 1)[0];
                self.data_indices.remove(sound1_idx);
                while True:
                    sound1, label1 = self.data[sound1_idx];
                    # sound1, label1 = self.data[random.randint(0, len(self.data) - 1)];
                    sound2, label2 = self.data[random.randint(0, len(self.data) - 1)];
                    if label1 != label2:
                        break;
                sound1 = self.preprocess(sound1);
                sound2 = self.preprocess(sound2);

                # Mix two examples
                r = np.array(random.random());
                sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32);
                eye = np.eye(self.opt.nClasses);
                label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32);

                #For stronger augmentation
                sound = U.random_gain(6)(sound).astype(np.float32);

            else:
                if indexes == None:
                    indexes = self.data[batchIndex*self.batch_size:(batchIndex+1)*self.batch_size];
                else:
                    if i >= len(indexes):
                        break;

                sound, target = indexes[i];
                sound = self.preprocess(sound).astype(np.float32);
                label = np.zeros(self.opt.nClasses);
                label[target] = 1;

            sounds.append(sound);
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);

        return sounds, labels;

    def preprocess_setup(self):
        funcs = [];
        if self.train:
            funcs += [U.random_scale(1.25), U.padding(self.opt.inputLength // 2), U.random_crop(self.opt.inputLength)];
        else:
            funcs += [U.match_length(self.opt.inputLength), U.fixed_crop(self.opt.inputLength)];

        funcs += [U.normalize(32768.0)];
        return funcs;

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound);

        return sound;


if __name__=='__main__':
    ds = 'iwingbeat';
    print('{} data generation has been started');
    opt = O.parse();
    O.display_info(opt);
    opt.nClasses= 10;
    opt.dataset = ds;
    opt.sr = 20000;
    opt.inputLength = 20000;
    opt.nEpochs = 600;

    mainDir = os.getcwd();
    data_dir = os.path.join(opt.data, '{}/data'.format(opt.dataset));
    if not os.path.exists(data_dir):
        os.mkdir(data_dir);

    #Generate Augmented Training Data
    data = np.load(os.path.join(opt.data, opt.dataset, 'raw-data/train.npz'), allow_pickle=True);
    X = data['x'];
    Y = data['y'];
    opt.batchSize=len(X);
    print('Train- BatchSize: {}, Epochs: {}'.format(opt.batchSize, opt.nEpochs));
    # aug_data_dir = os.path.join(data_dir, 'aug_data');
    # if not os.path.exists(aug_data_dir):
    #     os.mkdir(aug_data_dir);
    #
    # aug_data_dir = os.path.join(data_dir, 'aug_data', 'train');
    # if not os.path.exists(aug_data_dir):
    #     os.mkdir(aug_data_dir);
    #
    # trainGen = Generator(X, Y, opt, train=True);
    # for e in range(1, opt.nEpochs+1):
    #     start_time = time.perf_counter()
    #     start_time = time.perf_counter();
    #     trainX, trainY = trainGen.get_batch(0);
    #     trainGen.reset_indices();
    #     np.savez_compressed('{}/train{}'.format(aug_data_dir, e), x=trainX, y=trainY);
    #     print('Train-{} data with shape x{} and y{} took {:.2f} secs'.format(e, trainX.shape, trainY.shape, time.perf_counter()-start_time));
    #     sys.stdout.flush();

    #For training conventional classifiers
    trainGen = Generator(X, Y, opt, train=False);
    trainX, trainY = trainGen.get_batch(0);
    np.savez_compressed('{}/train'.format(data_dir), x=trainX, y=trainY);

    print('==========Finished training data generation================');

    #Generate Validation Data
    start_time = time.perf_counter();
    data = np.load(os.path.join(opt.data, opt.dataset, 'raw-data/val.npz'), allow_pickle=True);
    X = data['x'];
    Y = [int(idx) for idx in data['y']];
    opt.batchSize=len(X);
    print('Val batch size: {}'.format(opt.batchSize));
    valGen = Generator(X, Y, opt, train=False);
    valX, valY = valGen.get_batch(0);
    np.savez_compressed('{}/val'.format(data_dir), x=valX, y=valY);
    print('Validation data with shape x{} and y{} took {:.2f} secs'.format(valX.shape, valY.shape, time.perf_counter()-start_time));
    print('==========Finished validation data generation================');

    #Generate Test Data
    start_time = time.perf_counter();
    data = np.load(os.path.join(opt.data, opt.dataset, 'raw-data/test.npz'), allow_pickle=True);
    X = data['x'];
    Y = [int(idx) for idx in data['y']];
    opt.batchSize=len(X);
    print('Test batch size: {}'.format(opt.batchSize));
    testGen = Generator(X, Y, opt, train=False);
    testX, testY = testGen.get_batch(0);
    np.savez_compressed('{}/test'.format(data_dir), x=testX, y=testY);
    print('Test data with shape x{} and y{} took {:.2f} secs'.format(testX.shape, testY.shape, time.perf_counter()-start_time));
    print('==========Finished test data generation================');

    #Generate Test Data
    data = np.load(os.path.join(opt.data, opt.dataset, 'raw-data/unlblpool.npz'), allow_pickle=True);
    X = data['x'];
    Y = [int(idx) for idx in data['y']];
    opt.batchSize=len(X);
    print('Unlblpool batch size: {}'.format(opt.batchSize));
    unlblGen = Generator(X, Y, opt, train=False);
    unlblX, unlblY = unlblGen.get_batch(0);
    np.savez_compressed('{}/unlblpool'.format(data_dir), x=unlblX, y=unlblY);
    print('Unlblpool data with shape x{} and y{} took {:.2f} secs'.format(unlblX.shape, unlblY.shape, time.perf_counter()-start_time));
