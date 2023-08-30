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

sys.path.append(os.getcwd());
import resources.utils as U;
import resources.opts as opts;
import resources.models as models;
import resources.calculator as calc;

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
        self.testX = None;
        self.testY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;

    def Train(self):
        train_start_time = time.time();
        net = None;
        if opt.netSize == 'micro':
            net = models.GetMicroACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr).to(self.opt.device);
        else:
            net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr).to(self.opt.device);
        print('Model Loaded');
        calc.summary(net, (1,1,opt.inputLength));
        # exit();
        print("Training ACDNet on {} has been started".format(self.opt.dataset));

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(net.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);
        for epochIdx in range(1, self.opt.nEpochs+1):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            self.load_training_data(epochIdx);
            n_batches = math.ceil(len(self.trainX)/self.opt.batchSize);
            for batchIdx in range(n_batches):
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
            val_acc, val_loss = self.__validate(net, lossFunc);
            #Save best model
            self.__save_model(val_acc, epochIdx, net);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));

    def load_training_data(self, epoch):
        data = None;
        if self.opt.augmentData:
            data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/aug_data/train{}/train{}.npz'.format(self.opt.datasetSuffix, epoch)), allow_pickle=True);
        else:
            if self.trainX is None:
                data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/train.npz'), allow_pickle=True);

        if data is not None:
            self.trainX = torch.tensor(data['x'], dtype=torch.float32).to(self.opt.device);
            self.trainY = torch.tensor(data['y'], dtype=torch.float32).to(self.opt.device);
            print('Shapes - X: {}, Y:{}'.format(self.trainX.shape, self.trainY.shape));


    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val{}.npz'.format(self.opt.datasetSuffix)), allow_pickle=True);
        self.testX = torch.tensor(data['x']).to(self.opt.device);
        self.testY = torch.tensor(data['y']).to(self.opt.device);

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __get_batch(self, index):
        x = self.trainX[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        y = self.trainY[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        return x.to(self.opt.device), y.to(self.opt.device);

    def __validate(self, net, lossFunc):
        if self.testX is None:
            self.load_val_data();

        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = self.opt.batchSize;
            for idx in range(math.ceil(len(self.testX)/batch_size)):
                x = self.testX[idx*batch_size : (idx+1)*batch_size];
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, self.testY, lossFunc);
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
            fname = "{}/trained_models/{}_a{:.2f}_e{}.pt";
            old_model = fname.format(dir, self.opt.model_name.lower(), self.bestAcc, self.bestAccEpoch);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx;
            torch.save({'weight':net.state_dict(), 'config':net.ch_config}, fname.format(dir, self.opt.model_name.lower(), self.bestAcc, self.bestAccEpoch));

def Train(opt):
    print('Starting {} model Training'.format(opt.model_name.upper()));
    opts.display_info(opt);
    trainer = Trainer(opt);
    trainer.Train();

if __name__ == '__main__':
    dataset = 'iwingbeat'; #options: esc50, us8k, iwingbeat and ...
    opt = opts.parse();
    opt.netSize = 'micro'; #options: full, micro
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    opt.sr = 20000;
    opt.inputLength = 30225;
    opt.datasetSuffix = '';
    opt.dataset = dataset;
    opt.augmentData = True;
    if opt.dataset == 'esc50':
        opt.nEpochs = 1000;
        opt.model_name = 'esc50_acdnet';
        opt.nClasses = 50;
    elif opt.dataset == 'us8k':
        opt.datasetSuffix = ''; #empty or _small
        opt.nEpochs = 600;
        opt.model_name = 'us8k{}_acdnet'.format(opt.datasetSuffix);
        opt.nClasses = 10;
    elif opt.dataset == 'iwingbeat':
        opt.inputLength = 20000;
        opt.nEpochs = 600;
        opt.model_name = 'iwingbeat_acdnet';
        opt.nClasses = 10;
        opt.augmentData = False;
    else:
        print('Please select a dataset');
        exit();
    if opt.netSize == 'micro':
        opt.model_name = '{}_{}'.format(opt.netSize, opt.model_name);
    Train(opt);
