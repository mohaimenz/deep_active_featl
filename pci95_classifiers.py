import sys;
import os;
import glob;
import math;
import numpy as np;
import random;
import statistics as stat;
import time;
import torch;
import torch.optim as optim;
import pickle;
from copy import deepcopy;

sys.path.append(os.getcwd());
import resources.opts as opts;
import resources.models as models;
import resources.calculator as calc;
import resources.utils as U;
import resources.al_utils as alu;

class Tester:
    def __init__(self, opt=None):
        self.opt = opt;
        self.testX = None;
        self.testY = None;
        self.valX = None;
        self.valY = None;
        self.transformer = None;
        self.opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.load_model();
        self.load_val_data();
        self.load_test_data();

    def load_model(self):
        netType = 'micro_' if self.opt.netSize == 'micro' else '';
        net_path = os.path.join(self.opt.modelPath, '{}{}{}_acdnet_*.pt'.format(netType, self.opt.dataset, self.opt.datasetSuffix));
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.opt.device);
            net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
            net.load_state_dict(state['weight']);
            print('Base model loaded: {}'.format(file_paths[0]));
            self.transformer = alu.get_net(net, self.opt.nClasses*5);
        else:
            print('Model not found');
            exit();

    def transform_data(self, x):
        self.transformer.eval();
        features = None;
        with torch.no_grad():
            features = self.transformer(x).data.cpu().numpy();
        return features;

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        data_x = data['x'];
        trans_x = None;
        batch_size = self.opt.batchSize;
        for idx in range(math.ceil(len(data_x)/batch_size)):
            x = data_x[idx*batch_size : (idx+1)*batch_size];
            tx = self.transform_data(torch.tensor(x).to(self.opt.device));
            trans_x = tx if trans_x is None else np.concatenate((trans_x, tx));
        self.testX = trans_x;
        self.testY = np.argmax(np.array(data['y']), 1);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        data_x = data['x'];
        trans_x = None;
        batch_size = self.opt.batchSize;
        for idx in range(math.ceil(len(data_x)/batch_size)):
            x = data_x[idx*batch_size : (idx+1)*batch_size];
            tx = self.transform_data(torch.tensor(x).to(self.opt.device));
            trans_x = tx if trans_x is None else np.concatenate((trans_x, tx));
        self.valX = trans_x;
        self.valY = np.argmax(np.array(data['y']), 1);

    def TestClassifiers(self):
        start_time = time.time();
        dir = os.getcwd();
        model_path = '{}/trained_models/{}';

        for loop in range(0, self.opt.loops+1):
            start_time1 = time.time();
            # print(model_path.format(dir, 'al{}_{}{}_kncc_*.sav'.format(loop, self.opt.dataset, self.opt.datasetSuffix)))
            file_paths = glob.glob(model_path.format(dir, '{}al{}_{}{}_kncc_*.sav'.format(self.opt.netSize, loop, self.opt.dataset, self.opt.datasetSuffix)));
            # print(model_path.format(dir, '{}al{}_{}{}_kncc_*.sav'.format(self.opt.netSize, loop, self.opt.dataset, self.opt.datasetSuffix)));
            knnc = pickle.load(open(file_paths[0], 'rb'));
            print('Loaded: {}'.format(file_paths[0]))
            file_paths = glob.glob(model_path.format(dir, '{}al{}_{}{}_lgr_*.sav'.format(self.opt.netSize, loop, self.opt.dataset, self.opt.datasetSuffix)));
            lgr = pickle.load(open(file_paths[0], 'rb'));
            print('Loaded: {}'.format(file_paths[0]))
            file_paths = glob.glob(model_path.format(dir, '{}al{}_{}{}_ridge_*.sav'.format(self.opt.netSize, loop, self.opt.dataset, self.opt.datasetSuffix)));
            ridge = pickle.load(open(file_paths[0], 'rb'));
            print('Loaded: {}'.format(file_paths[0]))

            classifiers = [knnc, lgr, ridge];
            for clssifier in classifiers:
                #validation accuracy
                val_pred = clssifier.predict(self.valX);
                val_acc = (((val_pred==self.valY)*1).mean()*100).item();
                print('Val Acc {:.2f}'.format(val_acc));

                acc_log = [];
                sample_indices = list(range(0, len(self.testX)));
                iter = 0;
                while iter < 1000:
                    iter += 1;
                    #START: Bootstrap Sampling
                    bootstrap_indices = np.random.choice(sample_indices, replace=True, size=len(self.testX));
                    x = self.testX[bootstrap_indices];
                    y = self.testY[bootstrap_indices];
                    pred = clssifier.predict(x);
                    acc = (((pred==y)*1).mean()*100).item();
                    acc_log.append(acc);

                stdv = stat.stdev(acc_log);
                stdErr = stdv/math.sqrt(1000);
                mean = stat.mean(acc_log);
                lBound = mean - 1.96 * stdErr;
                uBound = mean + 1.96 * stdErr;
                print('{} - Classifier 95PCI: stdev {:.2f}, stderr: {:.2f}, mean {:.2f}, lBound {:.2f}, uBound {:.2f}'.format(self.opt.dataset, stdv, stdErr, mean, lBound, uBound));
            print("Elapsed time: {}".format(U.to_hms(time.time()-start_time1)));
        print("Elapsed time: {}".format(U.to_hms(time.time()-start_time)));


if __name__ == '__main__':
    opt = opts.parse();
    datasets = ['esc50', 'us8k', 'small', 'iwingbeat'];
    al_loops = [7, 15, 15, 20];
    classes = [50, 10, 10, 10];
    opt.sr = 20000;
    opt.modelPath = 'trained_models';
    opt.netSize = 'micro';
    opt.datasetSuffix = '';
    for idx, ds in enumerate(datasets):
        if idx in [0, 1, 2]:
            print('{}: Skipped'.format(ds));
            continue;
        if ds=='small':
            opt.dataset = 'us8k';
            opt.datasetSuffix = '_small';
        opt.dataset = ds;
        opt.inputLength = 20000 if ds == 'iwingbeat' else 30225;
        opt.nClasses = classes[idx];
        opt.loops = al_loops[idx];
        tester = Tester(opt);
        tester.TestClassifiers();
