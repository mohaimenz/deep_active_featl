import sys;
import os;
import glob;
import math;
import numpy as np;
from copy import deepcopy;
import glob;
import random;
import time;
import torch;
from sklearn.utils import shuffle;
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier;
import pickle;

sys.path.append(os.getcwd());
import resources.utils as U;
import resources.opts as opts;
import resources.models as models;
import resources.calculator as calc;
import resources.al_utils as alu;

#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################

class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.trainX = None;
        self.trainY = None;
        self.valX = None;
        self.valY = None;
        self.testX = None;
        self.testY = None;
        self.tuneX = None;
        self.tuneY = None;
        self.dataPoolX = None;
        self.dataPoolY = None;
        self.net = None;
        self.transformer = None;
        self.lblX = None;
        self.lblY = None;
        self.knncModel = None;
        self.lgrModel = None;
        self.ridgeModel = None;

    def load_acdnet_model(self):
        dir = os.getcwd();
        net_path = self.opt.modelPath;
        if self.opt.loopNo > 0:
            net_path = '{}al{}_{}_acdnet_*.pt'.format(self.opt.netSize, self.opt.loopNo, self.opt.dataset);

        net_path = 'trained_models/{}'.format(net_path);
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.opt.device);
            self.net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
            self.net.load_state_dict(state['weight']);
            if 'allblidx' in state:
                print('New Labels: {}'.format(len(state['allblidx'])));
                self.transfrom_labelled_data(state['allblidx']);
            print('Model Loaded from {}'.format(file_paths[0]));
            self.transformer = alu.get_net(deepcopy(self.net), self.opt.nClasses*5);
        else:
            print('Model not found');
            exit();

    def train(self):
        self.load_acdnet_model();

        if self.dataPoolX is None:
            self.load_train_data();
            self.load_data_pool();
            self.load_test_data();
            self.load_val_data();

        self.train_classifiers();

    def train_classifiers(self):
        if self.lblX is not None:
            trainX = np.concatenate((self.trainX, self.lblX), axis=0);
            trainY = np.concatenate((self.trainY, self.lblY), axis=0);
        else:
            trainX = self.trainX;
            trainY = self.trainY;

        print(trainX.shape);
        print(trainY.shape);
        self.knncModel = KNeighborsClassifier(n_neighbors=10)
        self.knncModel.fit(trainX, trainY);
        val_pred = self.knncModel.predict(self.valX);
        val_acc = (((val_pred==self.valY)*1).mean()*100).item();
        print('KNNC - val: {:.2f}'.format(val_acc));
        # if self.opt.nAlLoops == self.opt.loopNo:
        fname = "trained_models/{}al{}_{}{}_kncc_{:.2f}.sav";
        # pickle.dump(self.knncModel, open(fname.format(self.opt.netSize, self.opt.loopNo, self.opt.dataset, opt.datasetSuffix, val_acc), 'wb'));

        self.lgrModel = LogisticRegression(solver='lbfgs', max_iter=300);
        self.lgrModel.fit(trainX, trainY);
        val_pred = self.lgrModel.predict(self.valX);
        val_acc = (((val_pred==self.valY)*1).mean()*100).item();
        print('LGR - val: {:.2f}'.format(val_acc));
        # if self.opt.nAlLoops == self.opt.loopNo:
        fname = "trained_models/{}al{}_{}{}_lgr_{:.2f}.sav";
        # pickle.dump(self.lgrModel, open(fname.format(self.opt.netSize, self.opt.loopNo, self.opt.dataset, opt.datasetSuffix, val_acc), 'wb'));

        self.ridgeModel = RidgeClassifier(solver='auto', max_iter=200);
        self.ridgeModel.fit(trainX, trainY);
        val_pred = self.ridgeModel.predict(self.valX);
        val_acc = (((val_pred==self.valY)*1).mean()*100).item();
        print('Ridge - val: {:.2f}'.format(val_acc));
        # if self.opt.nAlLoops == self.opt.loopNo:
        fname = "trained_models/{}al{}_{}{}_ridge_{:.2f}.sav";
        # pickle.dump(self.ridgeModel, open(fname.format(self.opt.netSize, self.opt.loopNo, self.opt.dataset, opt.datasetSuffix, val_acc), 'wb'));

    def test_classifiers(self):
        val_pred = self.knncModel.predict(self.valX);
        val_acc1 = (((val_pred==self.valY)*1).mean()*100).item();
        test_pred = self.knncModel.predict(self.testX);
        test_acc1 = (((test_pred==self.testY)*1).mean()*100).item();

        val_pred = self.lgrModel.predict(self.valX);
        val_acc2 = (((val_pred==self.valY)*1).mean()*100).item();
        test_pred = self.lgrModel.predict(self.testX);
        test_acc2 = (((test_pred==self.testY)*1).mean()*100).item();

        val_pred = self.ridgeModel.predict(self.valX);
        val_acc3 = (((val_pred==self.valY)*1).mean()*100).item();
        test_pred = self.ridgeModel.predict(self.testX);
        test_acc3 = (((test_pred==self.testY)*1).mean()*100).item();

        return val_acc1, test_acc1, val_acc2, test_acc2, val_acc3, test_acc3;
        # return val_acc2, test_acc2;

    def load_data_pool(self):
        if self.dataPoolX is None:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/unlblpool{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
            self.dataPoolX = np.array(data['x']);
            self.dataPoolY = np.array(data['y']);

    def load_train_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/train{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        data_x = data['x'];
        self.trainY = np.argmax(data['y'], 1);

        trans_x = None;
        batch = self.opt.batchSize;
        for i in range(math.ceil(len(data_x)//batch)+1):
            tx = self.transform_data(torch.tensor(data_x[i*batch: (i+1)*batch], dtype=torch.float32).to(self.opt.device));
            trans_x = tx if trans_x is None else np.concatenate((trans_x, tx));
        self.trainX = trans_x;

    def transfrom_labelled_data(self, lblIdx):
        data_x = self.dataPoolX[lblIdx];
        data_y = self.dataPoolY[lblIdx];
        trans_x = None;
        batch = self.opt.batchSize;
        for i in range(math.ceil(len(data_x)//batch)+1):
            tx = self.transform_data(torch.tensor(data_x[i*batch: (i+1)*batch], dtype=torch.float32).to(self.opt.device));
            trans_x = tx if trans_x is None else np.concatenate((trans_x, tx));

        self.lblX = trans_x;
        self.lblY = np.argmax(data_y, 1);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        self.valX = self.transform_data(torch.tensor(data['x']).to(self.opt.device));
        self.valY = np.argmax(data['y'], 1);

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        trans_x = None;
        data_x = data['x'];
        batch = self.opt.batchSize;
        for i in range(math.ceil(len(data_x)//batch)+1):
            tx = self.transform_data(torch.tensor(data_x[i*batch: (i+1)*batch], dtype=torch.float32).to(self.opt.device));
            trans_x = tx if trans_x is None else np.concatenate((trans_x, tx));
        self.testX = trans_x;
        self.testY = np.argmax(data['y'], 1);

    def transform_data(self, x):
        self.transformer.eval();
        features = None;
        with torch.no_grad():
            features = self.transformer(x).data.cpu().numpy();
        return features;

    def deploy(self):
        v1, t1, v2, t2, v3, t3 = self.test_classifiers();
        # v2, t2 = self.test_classifiers();
        print('LOOP-{}'.format(self.opt.loopNo));
        print('\tKNNC - val: {:.2f}, test: {:.2f}'.format(v1, t1));
        print('\tLGR - val: {:.2f}, test: {:.2f}'.format(v2, t2));
        print('\tRIDGE - val: {:.2f}, test: {:.2f}'.format(v3, t3));

if __name__ == '__main__':
    datasets = ['esc50', 'us8k', '_small', 'iwingbeat'];
    opt = opts.parse();
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    opt.netSize = 'micro'; #options: full, micro
    opt.sr = 20000;
    opt.inputLength = 30225;
    for ds in datasets:
        if ds in  ['esc50', 'iwingbeat', '_small']:
            continue;

        if ds == '_small':
            opt.dataset = 'us8k';
        else:
            opt.dataset = ds;

        opt.datasetSuffix = '_small' if ds == '_small' else '';
        if opt.dataset == 'esc50':
            opt.nClasses = 50;
            opt.modelPath = 'esc50_acdnet_a60.00_e527.pt';
            opt.nAlLoops = 7;
        elif opt.dataset == 'us8k':
            opt.nClasses = 10;
            opt.nAlLoops = 15;
            if opt.datasetSuffix != '':
                opt.modelPath = 'us8k_small_acdnet_a69.46_e322.pt';
            else:
                # opt.modelPath = 'trained_models/us8k_acdnet_a87.74_e423.pt';
                opt.modelPath = 'micro_us8k_acdnet_a83.16_e573.pt';
        elif opt.dataset == 'iwingbeat':
            opt.nClasses = 10;
            opt.nAlLoops = 20;
            opt.inputLength = 20000;
            opt.modelPath = 'iwingbeat_acdnet_a66.28_e185.pt';
        else:
            print('Please select a dataset');
            exit();

        trainer = Trainer(opt);
        for i in range(0, opt.nAlLoops+1):
            trainer.opt.loopNo = i;
            trainer.train();
            trainer.deploy();
