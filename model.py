import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

torch.manual_seed(1)

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, num_mc):
        super(VAE, self).__init__()

        #parameter initialization
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.num_mc = num_mc

        #generative model for getting mu(X|Z) and diagonal entries of sigma(X|Z)
        self.gen = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU()
        )
        self.gen_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self.gen_std = nn.Sequential(
            nn.Linear(h_dim,x_dim),
            nn.Softplus()                           #Softplus is a smoothened version of ReLU
        )


        #Inference model for getting mu(Z|X) and diagonal entries of sigma(Z|X)
        self.inf = nn.Sequential(
            nn.Linear(x_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU()
        )
        self.inf_mean = nn.Sequential(
            nn.Linear(h_dim,z_dim),
            nn.Sigmoid()
        )
        self.inf_std = nn.Sequential(
            nn.Linear(h_dim,z_dim),
            nn.Softplus()
        )

        self.prior_z_mean = torch.zeros(self.z_dim)
        self.prior_z_std = torch.ones(self.z_dim)

    def forward(self, x):

        #encode
        hz_enc =  self.inf(x)
        z_mean = self.inf_mean(hz_enc)                  #determines q(Z|X) gaussian distribution
        z_std = self.inf_std(hz_enc)                    #determines q(Z|X) gaussian distribution
        kld = self._kld_gauss(z_mean, z_std, self.prior_z_mean, self.prior_z_std)
        nll = 0
        for i in range(self.num_mc):
            # Sample Z using the inference model (Monte Carlo sample)
            z_sample = self.sample_gaussian(z_mean, z_std)

            #decode the z_sample, generate x to ultimately get log(P(X|Z))
            hx_enc = self.gen(z_sample)
            x_mean = self.gen_mean(hx_enc)
            x_std = self.gen_std(hx_enc)
            nll = nll + self._nll_bernoulli(x_mean, x)
        nll = nll/(1.0 * self.num_mc)
        neg_elbo = kld + nll

        return z_sample, x_mean, neg_elbo, kld, nll,

    def sample_gaussian(self, mean, std):
        normalsample = torch.randn(std.size()[0]) #mean is a zero vector, all diagonals are 1 in std
        return normalsample.mul(std) + mean #squeeze the vector based on std



    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        pass