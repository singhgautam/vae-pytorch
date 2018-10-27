import torch
import torch.nn as nn
import torch.nn.functional as F
from variationalmodelbase import VariationalModelBase


class VariationalModelConvDeconvBase(VariationalModelBase):
    def __init__(self, h_dim, z_dim, encoder_class, decoder_class):
        self.h_dim = h_dim

        self.encoder_class = encoder_class
        self.decoder_class = decoder_class

        super(VariationalModelConvDeconvBase, self).__init__(z_dim)

        # generator
        self.gen = self.decoder_class(self.h_dim, self.z_dim)

        # inference
        self.inf = self.encoder_class(self.h_dim, self.z_dim)

    def forward(self, x_t, batch_size):
        # compute Q(Z|X) given context and x_t
        z_mean, z_logvar = self.inf(x_t)

        # compute prior over z_t given context
        z_mean_prior = torch.zeros(batch_size, self.z_dim).to(self.device)
        z_logvar_prior = torch.zeros(batch_size, self.z_dim).to(self.device)

        # get a sample of z_t
        z_sample = self.sample_gaussian(z_mean, (0.5 * z_logvar).exp(), batch_size)

        # get distribution over x_t given z_t and the context
        _x_t_mean = self.gen(z_sample)

        # compute kld, log-likelihood of x_t and sub-elbo
        kld = self._kld_gauss(z_mean, z_logvar, z_mean_prior, z_logvar_prior)
        nll = self._nll_bernoulli(_x_t_mean, x_t)
        elbo_t = - nll - kld

        return z_sample, _x_t_mean, elbo_t, kld, nll

    def sample_x_mean(self, batch_size):
        # compute prior over z_t given context
        z_mean_prior = torch.zeros(batch_size, self.z_dim).to(self.device)
        z_logvar_prior = torch.zeros(batch_size, self.z_dim).to(self.device)
        z_sample = self.sample_gaussian(z_mean_prior, (0.5 * z_logvar_prior).exp(), batch_size)
        x_gen_mean = self.gen(z_sample)
        return z_sample, x_gen_mean