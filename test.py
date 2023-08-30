import sys;
import os;
import glob;
import math;
import numpy as np;
import random;
import time;
import torch;
import torch.optim as optim;

import resources.opts as opts;
import resources.models as models;
import resources.calculator as calc;

class Tester:
    def __init__(self, opt=None):
        self.opt = opt;
        self.testX = None;
        self.testY = None;
        self.valX = None;
        self.valY = None;
        self.opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.load_test_data();
        self.load_val_data();

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/test/test.npz'), allow_pickle=True);
        self.testX = torch.tensor(data['x']).to(self.opt.device);
        self.testY = torch.tensor(data['y']).to(self.opt.device);
        print(self.testX.shape);

    def load_val_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'data/val/val.npz'), allow_pickle=True);
        self.valX = torch.tensor(data['x']).to(self.opt.device);
        self.valY = torch.tensor(data['y']).to(self.opt.device);
        print(self.valX.shape);

    def test_model(self, net, test=False):
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        if test:
            print('Testing');
            return self.__validate(net, lossFunc, self.testX, self.testY);
        else:
            print('Validating');
            return self.__validate(net, lossFunc, self.valX, self.valY);

    def __validate(self, net, lossFunc, dataX, dataY):
        print(dataX.shape);
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = self.opt.batchSize;
            for idx in range(math.ceil(len(dataX)/batch_size)):
                x = dataX[idx*batch_size : (idx+1)*batch_size];
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, dataY, lossFunc);
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

    def TestModel(self):
        dir = os.getcwd();
        net_path = self.opt.model_path;
        print(net_path)
        file_paths = glob.glob(net_path);
        for f in file_paths:
            state = torch.load(f, map_location=self.opt.device);
            config = state['config'];
            weight = state['weight'];
            # print(state['pruned_seq'])
            net = models.GetACDNetModel(self.opt.inputLength, self.opt.nClasses, self.opt.sr, config).to(self.opt.device);
            net.load_state_dict(weight);
            print('Model found at: {}'.format(f));
            val_acc, val_loss = self.test_model(net=net, test=False);
            print('Val: Loss {:.3f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));
            test_acc, test_loss = self.test_model(net=net, test=True);
            print('Test: Loss {:.3f}  Acc(top1) {:.2f}%'.format(test_loss, test_acc));



if __name__ == '__main__':
    opt = opts.parse();
    valid_path = False;
    model_path = 'trained_models/esc50_acdnet_a62.50_e807.pt';
    opt.model_path = glob.glob(os.path.join(os.getcwd(), model_path))[0];
    opt.inputLength = 30225;
    opt.sr = 20000;
    opt.nClasses = 50;
    tester = Tester(opt);
    tester.TestModel();
