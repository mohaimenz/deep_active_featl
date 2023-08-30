import sys;
import os;
import glob;
import math;
import numpy as np;
import glob;
import random;
import time;
import torch;
import torch.optim as optim;
from sklearn.utils import shuffle;
from sklearn.model_selection import train_test_split;

sys.path.append(os.getcwd());
import resources.utils as U;
import resources.opts as opts;
import resources.models as models;
import resources.calculator as calc;
import resources.data_gen_fly as flygen;


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
        self.tuneX = None;
        self.tuneY = None;
        self.trainX = None;
        self.trainY = None;
        self.valX = None;
        self.valY = None;
        self.testX = None;
        self.testY = None;
        self.dataPoolX = None;
        self.dataPoolY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;
        self.curEpoch = 0;
        self.changeIdx = [int(self.opt.nEpochs * i) for i in self.opt.schedule];
        self.lblIdx = None;
        self.iclDataGen = None;

    def Train(self):
        train_start_time = time.time();
        net = None;
        if self.opt.model_path != '':
            net_path = self.opt.model_path;
            print(self.opt.model_path)
            file_paths = glob.glob(net_path);
            if len(file_paths)>0 and os.path.isfile(file_paths[0]):
                state = torch.load(file_paths[0], map_location=self.opt.device);
                net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, channel_config=state['config']).to(self.opt.device);
                net.load_state_dict(state['weight']);
                print('Model Loaded from {}'.format(file_paths[0]));
            else:
                print('Model has not been found');
                exit();
        else:
            print('Please provide the path for the pretrained network');
            exit();

        # print(net);
        # calc.summary(net, (1,1,opt.inputLength));

        training_text = "Re-Training" if self.opt.retrain else "Training from Scratch";
        print("{} has been started. You will see update after finishing every training epoch and validation".format(training_text));

        #Freeze layers for the entire online training
        # net = self.freeze_layers(net);
        if self.dataPoolX is None:
            self.load_val_data();
            self.load_test_data();
            self.load_data_pool();

        self.label_new_batch();
        # exit();
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(net.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        #### Test the model with the data;
        net.eval();
        val_acc, val_loss = self.__validate(net, lossFunc, self.valX, self.valY);
        # self.__save_model(val_acc, 0, net);
        print('Val: Loss {:.3f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));

        test_acc, test_loss = self.__validate(net, lossFunc, self.testX, self.testY);
        # self.__save_model(val_acc, 0, net);
        print('Test: Loss {:.3f}  Acc(top1) {:.2f}%'.format(test_loss, test_acc));
        net.train();
        # exit();
        for epochIdx in range(1, self.opt.nEpochs+1):
            self.curEpoch = epochIdx;
            self.load_train_data(epochIdx);
            epoch_start_time = time.time();
            #Freeze and un-freeze layers following a schedule
            # if epochIdx in self.changeIdx:
            #     net = self.freeze_layers_scheduled(net);
            #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx);
            cur_lr = optimizer.param_groups[0]['lr'];

            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = math.ceil(len(self.trainX)/self.opt.batchSize);
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x = self.trainX[batchIdx*self.opt.batchSize: (batchIdx+1)*self.opt.batchSize];
                y = self.trainY[batchIdx*self.opt.batchSize: (batchIdx+1)*self.opt.batchSize];
                # zero the parameter gradients
                optimizer.zero_grad();

                # forward + backward + optimize
                outputs = net(x);
                running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                loss = lossFunc(outputs.log(), y);
                loss.backward();
                optimizer.step();

                running_loss += loss.item();

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            net.eval();
            val_acc, val_loss = self.__validate(net, lossFunc, self.valX, self.valY);
            #Save best model
            self.__save_model(val_acc, epochIdx, net);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            running_loss = 0;
            running_acc = 0;
            net.train();

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
        freeze_tfeb_upto = 24;
        freeze_sfeb = True;

        # print('SFEB FROZEN: {}'.format(freeze_sfeb));
        for p in net.sfeb.parameters():
            p.requires_grad = freeze_sfeb;

        for i, p in enumerate(net.tfeb.parameters()):
            # print(i);
            if i < freeze_tfeb_upto:
                # print('TFEB-{} Frozen'.format(i));
                p.requires_grad = False;
            else:
                # print('TFEB-{} OPEN'.format(i));
                p.requires_grad = True;

        return net;

    def label_new_batch(self):
        unlbl = np.arange(len(self.dataPoolX))[~self.lblIdx];
        q_idxs = random.sample(list(unlbl), k=100);
        self.tuneX = self.dataPoolX[q_idxs];
        self.tuneY = self.dataPoolY[q_idxs];
        self.lblIdx[q_idxs] = True;
        # print('Labeled Samples: {}'.format(q_idxs));
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'raw_data/unlblpool.npz'), allow_pickle=True);
        X = [x for idx, x in enumerate(data['x']) if idx in q_idxs];
        Y = [int(lbl) for idx, lbl in enumerate(data['y']) if idx in q_idxs];
        newOpt = deepcopy(self.opt);
        newOpt.batchSize = 100;
        self.iclDataGen = flygen.GenData(X, Y, newOpt, True);

    def load_train_data(self, epoch):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/train/train{}.npz'.format(epoch)), allow_pickle=True);
        indices = random.sample(list(range(len(data['x']))), 100);
        iclX, iclY = self.iclDataGen.get_batch(0);
        x = np.concatenate((iclX, data['x'][indices]), axis=0);
        y = np.concatenate((iclY, data['y'][indices]), axis=0);
        x, y = shuffle(x,y, random_state=42);
        self.trainX = torch.tensor(x, dtype=torch.float32).to(self.opt.device);
        self.trainY = torch.tensor(y, dtype=torch.float32).to(self.opt.device);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val/val.npz'), allow_pickle=True);
        self.valX = torch.tensor(data['x'], dtype=torch.float32).to(self.opt.device);
        self.valY = torch.tensor(data['y'], dtype=torch.float32).to(self.opt.device);

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test/test.npz'), allow_pickle=True);
        self.testX = torch.tensor(data['x'], dtype=torch.float32).to(self.opt.device);
        self.testY = torch.tensor(data['y'], dtype=torch.float32).to(self.opt.device);

    def load_data_pool(self):
        if self.dataPoolX is None:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/unlblpool/unlblpool.npz'), allow_pickle=True);
            self.dataPoolX = np.array(data['x']);
            self.dataPoolY = np.array(data['y']);
            if self.lblIdx is None:
                self.lblIdx = np.zeros(len(self.dataPoolX), dtype=bool);


    def __get_lr(self, epoch):
        divide_epoch = np.array(self.changeIdx);
        decay = sum(epoch >= divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __get_batch(self, index):
        x = self.trainX[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        y = self.trainY[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        return x.to(self.opt.device), y.to(self.opt.device);

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

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            epochIdx, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            dir = os.getcwd();
            fname = "icl{}_{}_a{:.2f}_e{}.pt";
            old_fname = fname.format(self.opt.loop_no, self.opt.model_name.lower(), self.bestAcc, self.bestAccEpoch);
            old_model = '{}/trained_models/{}'.format(dir, old_fname);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx;
            new_fname = fname.format(self.opt.loop_no, self.opt.model_name.lower(), self.bestAcc, self.bestAccEpoch);
            self.opt.model_path = 'trained_models/{}'.format(new_fname);
            torch.save({'weight':net.state_dict(), 'config':net.ch_config}, self.opt.model_path);

    def Test(self):
        path = glob.glob(os.path.join(os.getcwd(), self.opt.model_path))[0];
        state = torch.load(path, map_location=self.opt.device);
        config = state['config'];
        weight = state['weight'];
        net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, config).to(self.opt.device);
        net.load_state_dict(weight);
        print('Model found at: {}'.format(path));
        net.eval();
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        test_acc, test_loss = self.__validate(net, lossFunc, self.testX, self.testY);
        print('Test: Loss {:.3f}  Acc(top1) {:.2f}%'.format(test_loss, test_acc));

if __name__ == '__main__':
    dataset = 'us8k';
    start_time = time.time();
    opt = opts.parse();
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    opt.LR = 0.001; #The trained model had the highest accuracy on LR: 0.001
    opt.schedule = [0.15, 0.6, 0.9];
    opt.warmup = 0;
    opt.batchSize = 64;
    opt.sr = 20000;
    opt.inputLength = 30225;
    opt.nEpochs = 100;
    if dataset == 'esc50':
        opt.dataset = 'esc50';
        opt.model_name = 'nofrz_esc50_acdnet';
        opt.nClasses = 50;
        opt.nAlLoops = 10;
        opt.model_path = 'trained_models/esc50_acdnet_a60.00_e527.pt';
    elif dataset == 'us8k':
        opt.dataset = 'us8k';
        opt.model_name = 'nofrz_us8k_acdnet';
        opt.nClasses = 10;
        opt.nAlLoops = 20;
        opt.model_path = 'trained_models/us8k_acdnet_a87.74_e423.pt';
    else:
        print('Please select a dataset');
        exit();

    valX, valY, testX, testY, dataPoolX, dataPoolY, lblIdx = None, None, None, None, None, None, None;
    for i in range(1, opt.nAlLoops+1):
        opt.loop_no = i;
        trainer = Trainer(opt);
        if i>1:
            trainer.valX = valX;
            trainer.valY = valY;
            trainer.testX = testX;
            trainer.testY = testY;
            trainer.dataPoolX = dataPoolX;
            trainer.dataPoolY = dataPoolY;
            trainer.lblIdx = lblIdx;
        trainer.Train();
        trainer.Test();
        opt.model_path = trainer.opt.model_path;
        valX = trainer.valX;
        valY = trainer.valY;
        testX = trainer.testX;
        testY = trainer.testY;
        dataPoolX = trainer.dataPoolX;
        dataPoolY = trainer.dataPoolY;
        lblIdx = trainer.lblIdx;
        trainer = None;

    print('Execution finished in: {}'.format(U.to_hms(time.time()-start_time)));
