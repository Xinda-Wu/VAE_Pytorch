import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

num_epoch = 100
batch_size = 128
learning_rate = 1e-3
img_size = 784

# 设备配置
torch.cuda.set_device(0)  # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./conv_img'):
    os.mkdir("./conv_img")


def to_img(x):
    out = 0.5*(x+1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


img_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST(
    root='./data/MNIST',
    train=True,
    transform=img_transformer,
    download=False
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =======================Model=======================
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ======================= Loss & Optim =======================
model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# ======================= Train =======================
for epoch in range(num_epoch):
    for data in dataloader:
        img, _ = data
        # Cuda train
        img = Variable(img).to(device)
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ========== log ===========
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epoch, loss.item()))

    # ========== save img =======
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './conv_img/img_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
