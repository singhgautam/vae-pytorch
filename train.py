from model import VAE
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# initialize CUDA
print 'torch.version',torch.__version__
print 'torch.cuda.is_available()',torch.cuda.is_available()
device = torch.device("cuda") # other "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True}

def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        #Clean and normalize the data set between 0 and 1
        data = data.squeeze()
        data = (data - data.min())/(data.max() - data.min())
        data = data.view(-1,28*28)
        data = data.to(device)
        z_sample, x_mean, neg_elbo, kld, nll = model(data)
        neg_elbo.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if batch_idx%print_every==0:
            print 'EPOCH {} \t Batch Index {} \t KLDLoss {:.6f} \t NLLLoss {:.6f}'.format(
                epoch,
                batch_idx,
                kld,
                nll
            )
            x_sample = model.sample_x_mean()
            x_sample = x_sample.view(28, 28)
            plt.imsave('imsaves/imsave_epoch'+str(epoch)+'_batch'+str(batch_idx)+'.png', x_sample.cpu().detach().numpy(), cmap='gray')
        train_loss = train_loss + neg_elbo.item()
    print '==> TRAINING EPOCH {} Average Loss: {:.4f}'.format(
        epoch,
        train_loss/(len(train_loader.dataset))
    )

def test(epoch):
    mean_test_loss = 0
    mean_kld_loss = 0
    mean_nll_loss = 0
    for batch_idx, (data, _) in enumerate(test_loader):

        #Clean and normalize the data set between 0 and 1
        data = data.squeeze()
        data = (data - data.min())/(data.max() - data.min())
        data = data.view(-1,28*28)
        data = data.to(device)
        z_sample, x_mean, neg_elbo, kld, nll = model(data)
        mean_test_loss += neg_elbo.item()
        mean_kld_loss += kld
        mean_nll_loss += nll
    mean_test_loss /= len(test_loader.dataset)
    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)
    print '==> TESTING EPOCH {} \t Average Loss: {:.4f} \t Average KLD {:.6f} \t Average NLL {:.6f}'.format(
        epoch,
        mean_test_loss,
        mean_kld_loss,
        mean_nll_loss
    )

x_dim = 28*28
z_dim = 20
h_dim = 400
num_mc = 10
batch_size = 128
learning_rate = 1e-3
n_epochs = 100
save_every = 10
print_every = 10
clip = 10

model = VAE(x_dim, z_dim, h_dim, num_mc, device).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs
)

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)
    if epoch%save_every == 1:
        filename = 'modelsaves/vae_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), filename)
        print 'Model saved to',filename