from model import VAE
import torch

from torchvision import datasets, transforms
print 'torch.version',torch.__version__

x_dim = 10
z_dim = 5
h_dim = 10
num_mc = 10

model = VAE(x_dim, z_dim, h_dim, num_mc)

#Sample using simple gaussian
mean = torch.zeros(x_dim)
std = torch.ones(x_dim)
x_sample = model.sample_gaussian(mean, std)
print 'x_sample', x_sample

#Use the sample
z_sample, x_mean, neg_elbo, kld, nll = model.forward(x_sample)
print 'z_sample', z_sample
print 'x_mean', x_mean
print 'neg_elbo', neg_elbo
print 'kld', kld
print 'nll', nll
