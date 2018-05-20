from model import VAE
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
print 'torch.version',torch.__version__

x_dim = 28*28
z_dim = 20
h_dim = 40
num_mc = 10
batch_size = 128
learning_rate = 1e-3
n_epochs = 100
save_every = 10

model = VAE(x_dim, z_dim, h_dim, num_mc)
x_gen = model.sample_x_mean()
x_gen = x_gen.view(28,28)
plt.imsave('imsaves/sample_generated.png', x_gen.numpy(), cmap='gray')
