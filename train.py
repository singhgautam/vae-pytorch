from model import VAE
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
print 'torch.version',torch.__version__

def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.squeeze()
        # print 'data', data
        print 'data.size()', data.size()
        plt.imsave('imsaves/imsave_epoch'+str(epoch)+'_batch'+batch_idx+'.png', data[0].numpy(), cmap='gray')

x_dim = 10
z_dim = 5
h_dim = 10
num_mc = 10
batch_size = 128
learning_rate = 1e-3
n_epochs = 100
save_every = 10

model = VAE(x_dim, z_dim, h_dim, num_mc)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True
)

for epoch in range(1, n_epochs + 1):
    train(epoch)
    # test(epoch)
    if epoch%save_every == 1:
        filename = 'modelsaves/vae_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), filename)
        print 'Model saved to',filename