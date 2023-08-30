import os;
import glob;
import numpy as np;
import math;
from sklearn.metrics import pairwise_distances;
import pdb;
from scipy import stats;
from copy import deepcopy;
import torch;
import resources.models as models;

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X]);
    mu = [X[ind]];
    indsAll = [ind];
    centInds = [0.] * len(X);
    cent = 0;
    # print('#Samps\tTotal Distance');
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float);
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float);
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent;
                    D2[i] = newD[i];
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True);
        if sum(D2) == 0.0: pdb.set_trace();
        D2 = D2.ravel().astype(float);
        Ddist = (D2 ** 2)/ sum(D2 ** 2);
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist));
        ind = customDist.rvs(size=1)[0];
        while ind in indsAll: ind = customDist.rvs(size=1)[0];
        mu.append(X[ind]);
        indsAll.append(ind);
        cent += 1;
    return indsAll;

def get_net(net, emb_len):
    if emb_len == 512:
        # 512 FEATURES
        #Remove the last 8 layers from the end of tfeb to get (512, 4, 4) feature vector
        tfeb = list(net.tfeb.children())[:-8];
        #Add avgpool (4,4) and flatten to get 512 features
        tfeb.extend([torch.nn.AvgPool2d(kernel_size = (4,4))]);
    elif emb_len == 200:
        # 200 FEATURES
        #Remove the last 3 layers from the end of tfeb to get (50, 2, 2) feature vector
        tfeb = list(net.tfeb.children())[:-3];
    elif emb_len == 50 or emb_len == 250:
        # 50 FEATURES
        #Remove the last 2 layers from the end of tfeb to get (50, 1, 1) feature vector
        tfeb = list(net.tfeb.children())[:-2];

    #Flatten the vector
    tfeb.extend([torch.nn.Flatten()]);
    net.tfeb = torch.nn.Sequential(*tfeb);

    #Replacing softmax with Identity layer that basically returns the input. Use this to avoid breaking the model;
    net.output = models.Identity();
    return net;

def get_grad_embedding(net, opt, x, emb_len):
    net.eval();
    with torch.no_grad():
        out = None;
        batch_size = 128;
        for idx in range(math.ceil(len(x)/batch_size)):
            bx = x[idx*batch_size : (idx+1)*batch_size];
            scores = net(bx);
            out = scores.data if out is None else torch.cat((out, scores.data));

    batchProbs = out.data.cpu().numpy();
    maxInds = np.argmax(batchProbs,1);

    net_emb = deepcopy(net);
    # del net;
    net_emb = get_net(net_emb, emb_len);
    net_emb.eval();
    with torch.no_grad():
        e = None;
        batch_size = 128;
        for idx in range(math.ceil(len(x)/batch_size)):
            ebx = x[idx*batch_size : (idx+1)*batch_size];
            s = net_emb(ebx);
            e = s.data if e is None else torch.cat((e, s.data));

    emb = e.data.cpu().numpy();
    # del net_emb;

    embDim = emb_len;
    embedding = np.zeros([len(x), opt.nClasses * embDim]);
    for j in range(len(x)):
        for c in range(opt.nClasses):
            if c == maxInds[j]:
                embedding[j][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (1 - batchProbs[j][c])
            else:
                embedding[j][embDim * c : embDim * (c+1)] = deepcopy(emb[j]) * (-1 * batchProbs[j][c])
    return torch.Tensor(embedding);

def query(net, opt, x, batch_size):
    emb_len = 250 if opt.dataset=='esc50' else 50;
    gradEmbedding = get_grad_embedding(net, opt, x, emb_len).numpy();
    chosen = init_centers(gradEmbedding, batch_size);
    return chosen;
