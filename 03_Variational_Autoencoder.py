"""
__author__ = 'Xinda-Wu'
This is an improved implementation of the paper [Stochastic Gradient VB and the
Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad.
These changes make the network converge much faster.
"""

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image

epochs =10
num_epochs = 100
batch_size = 128
learning_rate = 1e-3
img_size = 784

# 设备配置
torch.cuda.set_device(0)  # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


# def to_img(x):
#     # out = 0.5*(x+1)
#     x = x.clamp(0, 1)
#     x = x.view(-1, 1, 28, 28)
#     return x


dataset = MNIST(
    root='./data/MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # 均值 向量
        self.fc22 = nn.Linear(400, 20)  # 保准方差 向量
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum', size_average=False)
    KLD = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    # # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)
    # # KL divergence
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        img, _ = data
        img = img.view(-1, img.size(0))
        img = Variable(img)
        img.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader),
                loss.item() / len(img)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_dataloader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_dataloader):
            img, _ = data
            img = img.view(-1, img.size(0))
            img = Variable(img)
            img.to(device)

            recon_batch, mu, logvar = model(img)
            test_loss += loss_function(recon_batch, img, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu().data,
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, 10 + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')



# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, data in enumerate(dataloader):
#         img, _ = data    # img, img_number_label
#         img = img.view(img.size(0), -1)
#         img = Variable(img)
#         if torch.cuda.is_available():
#             img = img.cuda()


#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(img)
#         loss = loss_function(recon_batch, img, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 batch_idx * len(img),
#                 len(dataloader.dataset), 100. * batch_idx / len(dataloader),
#                 loss.item() / len(img)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#         epoch, train_loss / len(dataloader.dataset)))
#     if epoch % 10 == 0:
#         save = to_img(recon_batch.cpu().data)
#         save_image(recon_batch.cpu().data, './vae_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './vae.pth')