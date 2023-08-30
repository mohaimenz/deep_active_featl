import sys;
import os;
import glob;
import math;
import numpy as np;
import random;
import statistics as stat;
import time;
import torch;
import pickle;
from copy import deepcopy;
import pandas as pd;

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
        self.opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.load_val_data();
        self.load_test_data();
        self.net = None;
        self.transformer = None;
        self.load_model("","");

    def load_model(self, model_type, loop):
        if model_type == "":
            netSize = 'micro_' if self.opt.netSize == 'micro' else '';
            net_path = os.path.join(self.opt.modelPath, '{}{}{}_acdnet_*.pt'.format(netSize, self.opt.dataset, self.opt.datasetSuffix));
            # print(net_path);
            # exit();
            file_paths = glob.glob(net_path);
            if len(file_paths)>0 and os.path.isfile(file_paths[0]):
                state = torch.load(file_paths[0], map_location=self.opt.device);
                self.net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
                self.net.load_state_dict(state['weight']);
                print('Base model loaded: {}'.format(file_paths[0]));
                self.transformer = alu.get_net(deepcopy(self.net), self.opt.nClasses*5);
            else:
                print('Model not found');
                exit();
        else:
            net_path = os.path.join(self.opt.modelPath, '{}{}{}_{}{}_acdnet_*.pt'.format(self.opt.netSize, model_type, loop, self.opt.dataset, self.opt.datasetSuffix));
            file_paths = glob.glob(net_path);
            if len(file_paths)>0 and os.path.isfile(file_paths[0]):
                state = torch.load(file_paths[0], map_location=self.opt.device);
                self.net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
                self.net.load_state_dict(state['weight']);
                print('Learned model loaded: {}'.format(file_paths[0]));
            else:
                print('Model not found');
                exit();

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        self.testX = np.array(data['x']);
        self.testY = np.array(data['y']);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        self.valX = torch.tensor(data['x']).to(self.opt.device);
        self.valY = torch.tensor(data['y']).to(self.opt.device);

    def test_val_data(self):
        acc = self.__validate(self.valX, self.valY);
        print('Validation accuracy: {:.2f}'.format(acc));

    def TestBaseModel(self):
        start_time = time.time();
        # self.load_model("", "");
        print('Val Acc {:.2f}'.format(self.__validate(self.valX, self.valY)));
        # continue;
        acc_log = [];
        sample_indices = list(range(0, len(self.testX)));
        iter = 0;
        while iter < 1000:
            iter += 1;
            #START: Bootstrap Sampling
            bootstrap_indices = np.random.choice(sample_indices, replace=True, size=len(self.testX));
            x = torch.tensor(self.testX[bootstrap_indices]).to(self.opt.device);
            y= torch.tensor(self.testY[bootstrap_indices]).to(self.opt.device);
            acc_log.append(self.__validate(x, y));

        stdv = stat.stdev(acc_log);
        mean = stat.mean(acc_log);
        lBound = mean - 1.96 * stdv;
        uBound = mean + 1.96 * stdv;
        print('{} - Base Model 95PCI: stdev {:.2f}, mean {:.2f}, lBound {:.2f}, uBound {:.2f}'.format(self.opt.dataset, stdv, mean, lBound, uBound));
        print("Elapsed time: {}".format(U.to_hms(time.time()-start_time)));
        return acc_log;

    def TestLearnedModel(self, type):
        start_time = time.time();
        self.load_model(type, self.opt.loops);
        print('Val Acc {:.2f}'.format(self.__validate(self.valX, self.valY)));
        # continue;
        acc_log = [];
        sample_indices = list(range(0, len(self.testX)));
        iter = 0;
        while iter < 1000:
            iter += 1;
            #START: Bootstrap Sampling
            bootstrap_indices = np.random.choice(sample_indices, replace=True, size=len(self.testX));
            x = torch.tensor(self.testX[bootstrap_indices]).to(self.opt.device);
            y= torch.tensor(self.testY[bootstrap_indices]).to(self.opt.device);
            acc_log.append(self.__validate(x, y));

        stdv = stat.stdev(acc_log);
        mean = stat.mean(acc_log);
        lBound = mean - 1.96 * stdv;
        uBound = mean + 1.96 * stdv;
        print('{} - Base Model 95PCI: stdev {:.2f}, mean {:.2f}, lBound {:.2f}, uBound {:.2f}'.format(self.opt.dataset, stdv, mean, lBound, uBound));
        print("Elapsed time: {}".format(U.to_hms(time.time()-start_time)));
        return acc_log;

    def transform_data(self, x):
        self.transformer.eval();
        features = None;
        with torch.no_grad():
            features = self.transformer(x).data.cpu().numpy();
        return features;

    def TestClassifiers(self):
        start_time = time.time();
        valY = np.argmax(np.array(self.valY.to('cpu')), 1);
        testY = np.argmax(self.testY, 1);

        # load the model from disk
        dir = os.getcwd();
        model_path = '{}/trained_models/{}';
        file_paths = glob.glob(model_path.format(dir, '{}al{}_{}{}_kncc_*.sav'.format(self.opt.netSize, self.opt.loops, self.opt.dataset, self.opt.datasetSuffix)));
        knnc = pickle.load(open(file_paths[0], 'rb'));
        file_paths = glob.glob(model_path.format(dir, '{}al{}_{}{}_lgr_*.sav'.format(self.opt.netSize, self.opt.loops, self.opt.dataset, self.opt.datasetSuffix)));
        lgr = pickle.load(open(file_paths[0], 'rb'));
        file_paths = glob.glob(model_path.format(dir, '{}al{}_{}{}_ridge_*.sav'.format(self.opt.netSize, self.opt.loops, self.opt.dataset, self.opt.datasetSuffix)));
        ridge = pickle.load(open(file_paths[0], 'rb'));

        result = [];
        classifiers = [knnc, lgr, ridge];
        for clssifier in classifiers:
            #validation accuracy
            val_pred = clssifier.predict(self.transform_data(self.valX));
            val_acc = (((val_pred==valY)*1).mean()*100).item();
            print('Val Acc {:.2f}'.format(val_acc));

            acc_log = [];
            sample_indices = list(range(0, len(self.testX)));
            iter = 0;
            while iter < 1000:
                iter += 1;
                #START: Bootstrap Sampling
                bootstrap_indices = np.random.choice(sample_indices, replace=True, size=len(self.testX));
                x = self.testX[bootstrap_indices];
                y = testY[bootstrap_indices];

                trans_x = None;
                batch = self.opt.batchSize;
                for i in range(math.ceil(len(x)//batch)+1):
                    tx =self.transform_data(torch.tensor(x[i*batch: (i+1)*batch], dtype=torch.float32).to(self.opt.device))
                    trans_x = tx if trans_x is None else np.concatenate((trans_x, tx));

                pred = clssifier.predict(trans_x);
                acc = (((pred==y)*1).mean()*100).item();
                acc_log.append(acc);

            stdv = stat.stdev(acc_log);
            stdErr = stdv/math.sqrt(1000);
            mean = stat.mean(acc_log);
            lBound = mean - 1.96 * stdErr;
            uBound = mean + 1.96 * stdErr;
            print('{} - Classifier 95PCI: stdev {:.2f}, mean {:.2f}, lBound {:.2f}, uBound {:.2f}'.format(self.opt.dataset, stdv, mean, lBound, uBound));

            result.append(acc_log);

        print("Elapsed time: {}".format(U.to_hms(time.time()-start_time)));
        return result;

    def __validate(self, dataX, dataY):
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        self.net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = self.opt.batchSize;
            for idx in range(math.ceil(len(dataX)/batch_size)):
                x = dataX[idx*batch_size : (idx+1)*batch_size];
                scores = self.net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, dataY, lossFunc);
        return acc;

    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            pred = y_pred.argmax(dim=1);
            target = y_target.argmax(dim=1);
            acc = (((pred==target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.log(), y_target).item();
            # loss = 0.0;
        return acc, loss;

if __name__ == '__main__':
    opt = opts.parse();
    datasets = ['esc50', 'us8k', 'small', 'iwingbeat'];
    al_loops = [7, 15, 15, 20];
    classes = [50, 10, 10, 10];
    opt.sr = 20000;
    opt.netSize = 'micro';
    opt.modelPath = 'trained_models';
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
        results = [];
        results.append(tester.TestBaseModel());
        results.append(tester.TestLearnedModel('icl'));
        results.append(tester.TestLearnedModel('al'));
        print('Classifiers Loading ....')
        results.extend(tester.TestClassifiers());
        cols = ['base', 'al_icl', 'al_featl', 'al_kncc', 'al_lgr', 'al_ridge'];
        df = pd.DataFrame(results, index=cols).T;
        df.to_csv('{}al_{}{}_1k_run.csv'.format(opt.netSize, opt.dataset, opt.datasetSuffix), index=False, sep=',');
        print('Completed writing into csv....');
