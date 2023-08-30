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
        self.train2X = None;
        self.train2Y = None;
        self.valX = None;
        self.valY = None;
        self.testX = None;
        self.testY = None;
        self.tuneX = None;
        self.tuneY = None;
        self.dataPoolX = None;
        self.dataPoolY = None;
        self.log = {'best_val_acc':0.0, 'best_val_acc_epoch':0, 'qbatch_acc1':0.0, 'qbatch_acc2':0.0};
        self.curEpoch = 0;
        self.changeIdx = [int(self.opt.nEpochs * i) for i in self.opt.schedule];
        self.fullNet = None;
        self.microNet = None;
        self.microLblIdx = None;
        self.fullLblIdx = None;

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

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(self.fullNet.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        lossFunc2 = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer2 = optim.SGD(self.microNet.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);


        self.fullNet.train();
        self.microNet.train();
        # exit();
        for epochIdx in range(1, self.opt.nEpochs+1):
            self.curEpoch = epochIdx;
            self.load_train_data(epochIdx);
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx);
            optimizer2.param_groups[0]['lr'] = self.__get_lr(epochIdx);

            n_batches = math.ceil(len(self.trainX)/self.opt.batchSize);
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x = self.trainX[batchIdx*self.opt.batchSize: (batchIdx+1)*self.opt.batchSize];
                y = self.trainY[batchIdx*self.opt.batchSize: (batchIdx+1)*self.opt.batchSize];
                # zero the parameter gradients
                optimizer.zero_grad();

                # forward + backward + optimize
                outputs = self.fullNet(x);
                loss = lossFunc(outputs.log(), y);
                loss.backward();
                optimizer.step();

                # with torch.no_grad():
                x2 = self.train2X[batchIdx*self.opt.batchSize: (batchIdx+1)*self.opt.batchSize];
                y2 = self.train2Y[batchIdx*self.opt.batchSize: (batchIdx+1)*self.opt.batchSize];
                optimizer2.zero_grad();
                # forward + backward + optimize
                outputs2 = self.microNet(x2);
                loss2 = lossFunc2(outputs2.log(), y2);
                loss2.backward();
                optimizer2.step();

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            self.fullNet.eval();
            val_acc, val_loss = self.__validate(self.fullNet, lossFunc, self.valX, self.valY);
            self.__save_model(val_acc, epochIdx, self.fullNet, 'full');


            self.microNet.eval();
            val_acc2, val_loss2 = self.__validate(self.microNet, lossFunc2, self.valX, self.valY);
            self.__save_model(val_acc2, epochIdx, self.microNet, 'micro');

            self.fullNet.train();
            self.microNet.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));

    def freeze_layers_scheduled(self, net):
        # print('EPOCH===={}'.format(self.curEpoch));
        tfeb_freeze_idx = [27, 21, 15, 7, 0]
        freeze_sfeb = True;
        freeze_tfeb_upto = tfeb_freeze_idx[0];

        if self.curEpoch in self.changeIdx:
            idx = self.changeIdx.index(self.curEpoch);
            # print(idx);
            freeze_tfeb_upto = tfeb_freeze_idx[idx+1];
            # print(freeze_tfeb_upto);

        if freeze_tfeb_upto==0:
            freeze_sfeb = False;

        # print('SFEB FROZEN: {}'.format(freeze_sfeb));
        for p in net.sfeb.parameters():
            p.requires_grad = freeze_sfeb;

        for i, p in enumerate(net.tfeb.parameters()):
            if i < freeze_tfeb_upto:
                # print('TFEB-{} Frozen'.format(i));
                p.requires_grad = False;
            else:
                # print('TFEB-{} OPEN'.format(i));
                p.requires_grad = True;

        return net;

    def freeze_layers(self, net):
        # print('EPOCH===={}'.format(self.curEpoch));
        freeze_tfeb_upto = 27;
        freeze_sfeb = True;

        # print('SFEB FROZEN: {}'.format(freeze_sfeb));
        for p in net.sfeb.parameters():
            p.requires_grad = freeze_sfeb;

        for i, p in enumerate(net.tfeb.parameters()):
            if i < freeze_tfeb_upto:
                # print('TFEB-{} Frozen'.format(i));
                p.requires_grad = False;
            else:
                # print('TFEB-{} OPEN'.format(i));
                p.requires_grad = True;

        return net;

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
        self.tuneX = self.dataPoolX[fq_idxs];
        self.tuneY = self.dataPoolY[fq_idxs];

        #For micro net
        unlbl = np.arange(len(self.dataPoolX))[~self.microLblIdx]
        x = self.dataPoolX[unlbl];
        y = self.dataPoolY[unlbl];
        x = torch.tensor(x, dtype=torch.float32).to(self.opt.device);
        y = torch.tensor(y, dtype=torch.float32).to(self.opt.device);

        chosen = alu.query(deepcopy(self.microNet), self.opt, x, self.opt.newLabelsPerLoop);
        mq_idxs = unlbl[chosen];
        self.microLblIdx[mq_idxs] = True;
        self.tune2X = self.dataPoolX[mq_idxs];
        self.tune2Y = self.dataPoolY[mq_idxs];
        matches = set(fq_idxs).intersection(mq_idxs);
        print('Matches: {}'.format(matches));
        print('Match Count: {}'.format(len(list(matches))));

    def load_data_pool(self):
        if self.dataPoolX is None:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/unlblpool{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
            self.dataPoolX = np.array(data['x']);
            self.dataPoolY = np.array(data['y']);
            if self.fullLblIdx is None:
                self.fullLblIdx = np.zeros(len(self.dataPoolX), dtype=bool);
                self.microLblIdx = np.zeros(len(self.dataPoolX), dtype=bool);

    def load_train_data(self, epoch):
        data = None;
        if self.opt.augmentData:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/aug_data/train{}/train{}.npz'.format(self.opt.datasetSuffix, epoch)), allow_pickle=True);
        else:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/train.npz'), allow_pickle=True);

        indices = random.sample(list(range(len(data['x']))), int(self.opt.newLabelsPerLoop * self.opt.oldNewDataRatio));
        x = np.concatenate((self.tuneX, data['x'][indices]), axis=0);
        y = np.concatenate((self.tuneY, data['y'][indices]), axis=0);
        x, y = shuffle(x,y, random_state=42);
        self.trainX = torch.tensor(x, dtype=torch.float32).to(self.opt.device);
        self.trainY = torch.tensor(y, dtype=torch.float32).to(self.opt.device);

        x2 = np.concatenate((self.tune2X, data['x'][indices]), axis=0);
        y2 = np.concatenate((self.tune2Y, data['y'][indices]), axis=0);
        x2, y2 = shuffle(x2,y2, random_state=42);
        self.train2X = torch.tensor(x2, dtype=torch.float32).to(self.opt.device);
        self.train2Y = torch.tensor(y2, dtype=torch.float32).to(self.opt.device);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        self.valX = torch.tensor(data['x']).to(self.opt.device);
        self.valY = torch.tensor(data['y']).to(self.opt.device);

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        self.testX = torch.tensor(data['x']).to(self.opt.device);
        self.testY = torch.tensor(data['y']).to(self.opt.device);

    def __get_lr(self, epoch):
        divide_epoch = np.array(self.changeIdx);
        decay = sum(epoch >= divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __validate(self, net, lossFunc, testX, testY):
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = self.opt.batchSize;
            for idx in range(math.ceil(len(testX)/batch_size)):
                x = testX[idx*batch_size : (idx+1)*batch_size];
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, testY, lossFunc);
        net.train();
        return acc, loss;

    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            pred = y_pred.argmax(dim=1);
            target = y_target.argmax(dim=1);
            acc = (((pred==target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.log(), y_target).item();
            # loss = 0.0;
        return acc, loss;

    def __save_model(self, acc, epochIdx, net, modelType):
        modelName = self.opt.modelNameFull if modelType == 'full' else self.opt.modelNameMicro;
        if acc > self.log['best_val_acc']:
            dir = os.getcwd();
            fname = "{}al{}_{}_a{:.2f}_e{}.pt";
            old_fname = fname.format(self.opt.netSize, self.opt.loopNo, modelName.lower(), self.log['best_val_acc'], self.log['best_val_acc_epoch']);
            old_model = '{}/trained_models/full_micro/{}'.format(dir, old_fname);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.log['best_val_acc'] = acc;
            self.log['best_val_acc_epoch'] = epochIdx;
            new_fname = fname.format(self.opt.netSize, self.opt.loopNo, modelName.lower(), self.log['best_val_acc'], self.log['best_val_acc_epoch']);
            if modelType == 'full':
                self.opt.fullModelPath = 'trained_models/full_micro/{}'.format(new_fname);
            else:
                self.opt.microModelPath = 'trained_models/full_micro/{}'.format(new_fname);

            modelPath = self.opt.fullModelPath if modelType == 'full' else self.opt.microModelPath;
            torch.save({'weight':net.state_dict(), 'config':net.ch_config}, modelPath);

    def deploy(self):
        path = glob.glob(os.path.join(os.getcwd(), self.opt.fullModelPath))[0];
        state = torch.load(path, map_location=self.opt.device);
        config = state['config'];
        weight = state['weight'];
        net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, config).to(self.opt.device);
        net.load_state_dict(weight);
        print('Model found at: {}'.format(path));

        path2 = glob.glob(os.path.join(os.getcwd(), self.opt.microModelPath))[0];
        state2 = torch.load(path2, map_location=self.opt.device);
        config2 = state2['config'];
        weight2 = state2['weight'];
        net2 = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, config2).to(self.opt.device);
        net2.load_state_dict(weight2);
        print('Model found at: {}'.format(path2));


        net.eval();
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        self.log['test_acc'], test_loss = self.__validate(net, lossFunc, self.testX, self.testY);
        print('Full Model - loop{} - Test Acc: {:.2f}'.format(self.opt.loopNo, self.log['test_acc']));

        net2.eval();
        self.log['test_acc2'], test_loss = self.__validate(net2, lossFunc, self.testX, self.testY);
        print('Micro Model - loop{} - Test Acc: {:.2f}'.format(self.opt.loopNo, self.log['test_acc2']));

if __name__ == '__main__':
    dataset = 'esc50'; #options: esc50, us8k, iwingbeat and ...
    start_time = time.time();
    opt = opts.parse();
    opt.netSize = 'micro'; #options: full, micro
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    opt.initLR = 0.001; #The trained model had the highest accuracy on LR: 0.001
    opt.schedule = [0.3, 0.6, 0.9];
    opt.warmup = 0;
    opt.batchSize = 16;
    opt.sr = 20000;
    opt.inputLength = 30225;
    opt.nEpochs = 100;
    opt.datasetSuffix = '';
    opt.dataset = dataset;
    opt.augmentData = True;
    opt.newLabelsPerLoop = 100;
    opt.oldNewDataRatio = 1;
    if opt.dataset == 'esc50':
        opt.modelNameFull = 'full_esc50';
        opt.modelNameMicro = 'micro_esc50';
        opt.nClasses = 50;
        opt.nAlLoops = 7;
        opt.fullModelPath = 'trained_models/full_micro/esc50_acdnet_a60.00_e527.pt';
        opt.microModelPath = 'trained_models/full_micro/micro_esc50_acdnet_a59.50_e827.pt';
    elif opt.dataset == 'us8k':
        opt.datasetSuffix = ''; #Options: _small, empty ('')
        opt.modelName = 'us8k{}_acdnet'.format(opt.datasetSuffix);
        opt.nClasses = 10;
        opt.nAlLoops = 15;
        if opt.datasetSuffix != '':
            opt.modelPath = 'trained_models/us8k_small_acdnet_a69.46_e322.pt';
        else:
            # opt.modelPath = 'trained_models/us8k_acdnet_a87.74_e423.pt';
            opt.modelPath = 'trained_models/micro_us8k_acdnet_a83.16_e573.pt';
    elif opt.dataset == 'iwingbeat':
        opt.inputLength = 20000;
        opt.batchSize = 32;
        opt.nClasses = 10;
        opt.nAlLoops = 20;
        # opt.nEpochs = 50;
        opt.newLabelsPerLoop = 500;
        opt.oldNewDataRatio = 1;
        opt.augmentData = False;
        opt.modelName = 'iwingbeat_acdnet';
        # opt.modelPath = 'trained_models/iwingbeat_acdnet_a66.28_e185.pt';
        opt.modelPath = 'trained_models/micro_iwingbeat_acdnet_a63.96_e516.pt';
    else:
        print('Please select a dataset');
        exit();

    valX, valY, testX, testY, dataPoolX, dataPoolY, fullLblIdx, microLblIdx = None, None, None, None, None, None, None, None;
    for i in range(1, opt.nAlLoops+1):
        opt.LR = opt.initLR;
        opt.loopNo = i;
        trainer = Trainer(opt);
        if i>1:
            trainer.opt.LR = opt.initLR / 2 if i > int(opt.nAlLoops//2) else opt.initLR;
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
