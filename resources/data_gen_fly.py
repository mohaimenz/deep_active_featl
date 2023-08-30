import os;
import sys;
import numpy as np;
import random;
import time;
from sklearn.utils import shuffle;
from sklearn.model_selection import train_test_split;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'resources'));
import utils as U;
import opts as O;

class GenData():
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
        self.reset_indices();
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
