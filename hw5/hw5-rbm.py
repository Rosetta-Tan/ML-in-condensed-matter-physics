"""
We thank bacnguyencong for paving the basis of this example code.

Restricted Boltzmann machine (RBM) for MNIST.
Please run the program, tune parameters and plot the figures.
Submit the runtime outputs and the best figures on the course.pku.edu.cn.
Note that you need to submit a total of 3 figures.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


epochs, batchsize = 100, 64
n_vis, n_hid, k = 784 + 10, 256, 1  # parameters of RBM, k is the times of Gibbs sampling.
lr, wd, betas = 1e-3, 0., (0.9, 0.999)  # parameters of optimizer
ss, gamma = 50, 0.5  # parameters of lr_scheduler
noise, mask_rate = 0.25, 0.25 # parameters of reconstruction

np.random.seed(100)
torch.manual_seed(0)


class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=128):
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_vis))
        self.h = nn.Parameter(torch.randn(1, n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))

    def visible_to_hidden(self, v):
        p = torch.sigmoid(F.linear(v, self.W, self.h))
        return p, p.bernoulli()

    def hidden_to_visible(self, h):
        p = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        return p, p.bernoulli()

    def energy(self, v):
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, v, k=1):
        for _ in range(k):
            h = self.visible_to_hidden(v)[1]
            logits, v = self.hidden_to_visible(h)
        return logits, v


def plotfig(img, filename):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.imsave("./{}.png".format(filename), npimg)
    plt.show()
    plt.close()


if __name__ == "__main__":
    if not os.path.exists('./data_rbm'):
        os.mkdir('./data_rbm')

    model = RBM(n_vis=n_vis, n_hid=n_hid)

    trainset = datasets.MNIST('./data_rbm', train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    testset = datasets.MNIST('./data_rbm', train=False, download=True,
                             transform=transforms.Compose([transforms.ToTensor()]))
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=ss, gamma=gamma)

    # train
    model.train()
    losslist = []
    for epoch in range(epochs):
        epoch_loss = 0.
        for data, labels in trainloader:
            data, labels = torch.round(data.view(-1, 784)), F.one_hot(labels, num_classes=10).float()
            v = torch.cat((labels, data), dim=1)
            v_gibbs = model(v.clone(), k=k)[1]
            loss = model.energy(v) - model.energy(v_gibbs)
            epoch_loss += loss.item() * len(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        losslist.append(epoch_loss / len(trainset))
        print('epoch={}\tloss={}'.format(epoch, losslist[-1]))
        # save model
        torch.save(model, './data_rbm/rbm.pkl')

    plt.figure()
    plt.plot(losslist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./data_rbm/Loss.png')
    plt.show()
    plt.close()

    # test
    # model = torch.load('./data_rbm/rbm.pkl')
    model.eval()

    # Image recognition, which is equivalent to the reconstruction of the first 10 elements.
    classes = np.arange(10)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, labels in testloader:
            data, length = torch.round(data.view(-1, 784)), len(labels)
            logits, v_gibbs = model(torch.cat((torch.zeros(length, 10), data), dim=1), k=1)
            for _ in range(1, 100):
                v_gibbs[:, 10:] = data
                logits, v_gibbs = model(v_gibbs, k=1)
            predicted = torch.max(logits[:, :10], 1)[1]
            c = (predicted == labels).squeeze()
            for i in range(length):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy: {:.3f} %'.format(100 * sum(class_correct) / sum(class_total)))
    for i in range(10):
        print('Accuracy of {:d} : {:.3f} %'.format(classes[i], 100 * class_correct[i] / class_total[i]))

    # Generating handwritten digits
    amount = 5
    labels = torch.repeat_interleave(torch.arange(10), amount)
    label_vectors = F.one_hot(labels, num_classes=10)
    genImage = model(torch.cat((label_vectors, torch.randint(2, (len(labels), 784)).float()), dim=1), k=1)[1]
    for _ in range(1, 10000):
        genImage[:, :10] = label_vectors
        genImage = model(genImage, k=1)[1]
    plotfig(make_grid(genImage[:, 10:].view(len(labels), 1, 28, 28).data, nrow=5, pad_value=1), './data_rbm/genImage')
    # # mean
    # genImage = genImage[:, 10:].view(10, amount, 784).data
    # genImage = torch.mean(genImage, dim=1)
    # plotfig(make_grid(genImage.view(10, 1, 28, 28).data, nrow=5, pad_value=1), './data_rbm/genImage_mean')

    # Reconstruction
    images, labels = next(iter(trainloader))
    images, labels = torch.round(images.view(-1, 784)), F.one_hot(labels, num_classes=10)
    plotfig(make_grid(images.view(batchsize, 1, 28, 28), pad_value=1), './data_rbm/original')
    v_gibbs = model(torch.cat((labels, images), dim=1), k=1)[1]
    plotfig(make_grid(v_gibbs[:, 10:].view(batchsize, 1, 28, 28).data, pad_value=1), './data_rbm/recon')
    # random_noise
    random_noise = (2 * torch.rand(batchsize, 784) - 1) * noise
    images_ = torch.clamp(images + random_noise, max=1, min=0)
    plotfig(make_grid(images_.view(batchsize, 1, 28, 28), pad_value=1), './data_rbm/original_noise')
    v_gibbs = model(torch.cat((labels, images_), dim=1), k=1)[1]
    plotfig(make_grid(v_gibbs[:, 10:].view(batchsize, 1, 28, 28).data, pad_value=1), './data_rbm/recon_noise')
    # mask
    mask = torch.from_numpy(np.random.choice(range(2), size=(batchsize, 784), p=[1 - mask_rate, mask_rate])).float()
    images_ = torch.clamp(images - mask, max=1, min=0)
    plotfig(make_grid(images_.view(batchsize, 1, 28, 28), pad_value=1), './data_rbm/original_mask')
    v_gibbs = model(torch.cat((labels, images_), dim=1), k=1)[1]
    plotfig(make_grid(v_gibbs[:, 10:].view(batchsize, 1, 28, 28).data, pad_value=1), './data_rbm/recon_mask')
