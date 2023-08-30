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
import torch.optim as optim;
from sklearn.utils import shuffle;

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
        self.fullNet = None;
        self.microNet = None;
        self.fullLblIdx = None;
        self.microLblIdx = None;

    def load_model(self):
        dir = os.getcwd();
        net_path = self.opt.fullModelPath;
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.opt.device);
            self.fullNet = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
            self.fullNet.load_state_dict(state['weight']);
            print('Full model loaded from {}'.format(file_paths[0]));
        else:
            print('Full model not found');
            exit();

        # calc.summary(self.fullNet, (1,1,opt.inputLength));

        net_path = self.opt.microModelPath;
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.opt.device);
            self.microNet = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
            self.microNet.load_state_dict(state['weight']);
            print('Micro model loaded from {}'.format(file_paths[0]));
        else:
            print('Micro model not found');
            exit();

        # calc.summary(self.microNet, (1,1,opt.inputLength));


    def train(self):
        train_start_time = time.time();
        print("Online training of ACDNet for fine-tuning is starting");

        #Freeze layers for the entire online training
        # net = self.freeze_layers(net);
        if self.dataPoolX is None:
            self.load_data_pool();
            self.load_test_data();
            self.load_val_data();

        self.load_model();
        self.label_batch();

        self.microNet.eval();
        val_acc = self.__validate(self.microNet, self.valX, self.valY);
        self.fullNet.eval();
        val_acc = self.__validate(self.fullNet, self.valX, self.valY);

        opt.fullModelPath = 'trained_models/fullal/al{}_esc50_acdnet_*.pt'.format(self.opt.loopNo);
        opt.microModelPath = 'trained_models/microal/microal{}_esc50_acdnet_*.pt'.format(self.opt.loopNo);

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));

    def label_batch(self):
        #For full net
        unlbl = np.arange(len(self.dataPoolX))[~self.fullLblIdx]
        x = self.dataPoolX[unlbl];
        y = self.dataPoolY[unlbl];
        x = torch.tensor(x, dtype=torch.float32).to(self.opt.device);
        y = torch.tensor(y, dtype=torch.float32).to(self.opt.device);

        chosen = alu.query(deepcopy(self.fullNet), self.opt, x, self.opt.newLabelsPerLoop);
        fq_idxs = unlbl[chosen];
        self.fullLblIdx[fq_idxs] = True;

        #For micro net
        unlbl = np.arange(len(self.dataPoolX))[~self.microLblIdx]
        x = self.dataPoolX[unlbl];
        y = self.dataPoolY[unlbl];
        x = torch.tensor(x, dtype=torch.float32).to(self.opt.device);
        y = torch.tensor(y, dtype=torch.float32).to(self.opt.device);

        chosen = alu.query(deepcopy(self.microNet), self.opt, x, self.opt.newLabelsPerLoop);
        mq_idxs = unlbl[chosen];
        self.microLblIdx[mq_idxs] = True;
        matches = set(fq_idxs).intersection(mq_idxs);
        print('Matches: {}'.format(matches));
        print('Match Count: {}'.format(len(list(matches))));

    def load_data_pool(self):
        if self.dataPoolX is None:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/unlblpool.npz'), allow_pickle=True);
            self.dataPoolX = np.array(data['x']);
            self.dataPoolY = np.array(data['y']);
            if self.fullLblIdx is None:
                self.fullLblIdx = np.zeros(len(self.dataPoolX), dtype=bool);
                self.microLblIdx = np.zeros(len(self.dataPoolX), dtype=bool);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val.npz'), allow_pickle=True);
        self.valX = torch.tensor(data['x']).to(self.opt.device);
        self.valY = torch.tensor(data['y']).to(self.opt.device);

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test.npz'), allow_pickle=True);
        self.testX = torch.tensor(data['x']).to(self.opt.device);
        self.testY = torch.tensor(data['y']).to(self.opt.device);

    def __validate(self, net, testX, testY):
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = self.opt.batchSize;
            for idx in range(math.ceil(len(testX)/batch_size)):
                x = testX[idx*batch_size : (idx+1)*batch_size];
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc = self.__compute_accuracy(y_pred, testY);
        return acc

    def __compute_accuracy(self, y_pred, y_target):
        with torch.no_grad():
            pred = y_pred.argmax(dim=1);
            target = y_target.argmax(dim=1);
            acc = (((pred==target)*1).float().mean()*100).item();
        return acc;

    def deploy(self):
        path = glob.glob(os.path.join(os.getcwd(), self.opt.fullModelPath))[0];
        state = torch.load(path, map_location=self.opt.device);
        config = state['config'];
        weight = state['weight'];
        net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, config).to(self.opt.device);
        net.load_state_dict(weight);
        print('Model found at: {}'.format(path));
        # net.eval();
        # print('Learned model deployed on {} samples'.format(len(self.testX)));
        # acc = self.__validate(net, self.testX, self.testY);
        # print('AL loop{}: Full Model - Test Acc: {:.2f}'.format(self.opt.loopNo, acc));

if __name__ == '__main__':
    dataset = 'esc50'; #options: esc50, us8k, iwingbeat and ...
    start_time = time.time();
    opt = opts.parse();
    opt.netSize = 'micro'; #options: full, micro
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    opt.batchSize = 16;
    opt.sr = 20000;
    opt.inputLength = 30225;
    opt.dataset = dataset;
    opt.newLabelsPerLoop = 100;
    if opt.dataset == 'esc50':
        opt.nClasses = 50;
        opt.nAlLoops = 7;
        # opt.modelPath = 'trained_models/esc50_acdnet_a60.00_e527.pt';
        opt.fullModelPath = 'trained_models/fullal/esc50_acdnet_a60.00_e527.pt';
        opt.microModelPath = 'trained_models/microal/micro_esc50_acdnet_a59.50_e827.pt';
    elif opt.dataset == 'us8k':
        opt.nClasses = 10;
        opt.nAlLoops = 15;
        opt.modelPath = 'trained_models/micro_us8k_acdnet_a83.16_e573.pt';
    elif opt.dataset == 'iwingbeat':
        opt.inputLength = 20000;
        opt.batchSize = 32;
        opt.nClasses = 10;
        opt.nAlLoops = 20;
        opt.newLabelsPerLoop = 500;
        opt.modelPath = 'trained_models/micro_iwingbeat_acdnet_a63.96_e516.pt';
    else:
        print('Please select a dataset');
        exit();

    valX, valY, testX, testY, dataPoolX, dataPoolY, fullLblIdx, microLblIdx = None, None, None, None, None, None, None, None;
    for i in range(1, opt.nAlLoops+1):
        opt.loopNo = i;
        trainer = Trainer(opt);
        if i>1:
            trainer.valX = valX;
            trainer.valY = valY;
            trainer.testX = testX;
            trainer.testY = testY;
            trainer.dataPoolX = dataPoolX;
            trainer.dataPoolY = dataPoolY;
            trainer.fullLblIdx = fullLblIdx;
            trainer.microLblIdx = microLblIdx;
        trainer.train();
        trainer.deploy();
        opt.fullModelPath = trainer.opt.fullModelPath;
        opt.microModelPath = trainer.opt.microModelPath;
        valX = trainer.valX;
        valY = trainer.valY;
        testX = trainer.testX;
        testY = trainer.testY;
        dataPoolX = trainer.dataPoolX;
        dataPoolY = trainer.dataPoolY;
        fullLblIdx = trainer.fullLblIdx;
        microLblIdx = trainer.microLblIdx;
        trainer = None;

    print('Execution finished in: {}'.format(U.to_hms(time.time()-start_time)));
