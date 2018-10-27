from variationalmodels.variationalmodel_convdeconv_omniglot import VariationalModelConvDeconvOmniglot
import torch
import torchvision
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
        sample_dataset_img = data[0]
        # print('sample_dataset_img.size()', sample_dataset_img.size())
        # print('sample_dataset_img.max()', sample_dataset_img.max().item())
        data = data.view(-1,3*105*105)
        data = data.to(device)
        # torchvision.utils.save_image(sample_dataset_img.cpu().detach(),
        #                              'imsaves_shepardmetzlar/im_sample_' + str(epoch) + '_batch' + str(batch_idx) + '.png')

        z_sample, x_mean, elbo, kld, nll = model(data, batch_size)

        # mean ELBO for the entire batch
        mean_neg_elbo = -elbo.mean()
        mean_neg_elbo.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if batch_idx%print_every==0:
            print 'EPOCH {} \t Batch Index {} \t KLDLoss {:.6f} \t NLLLoss {:.6f}'.format(
                epoch,
                batch_idx,
                kld.mean(),
                nll.mean()
            )
            z_sample, x_sample = model.sample_x_mean(1)
            x_sample = x_sample.view(3, 105, 105)
            # print('save_img.size()', x_sample.size())
            # print('save_dataset_img.max()', x_sample.max().item())
            torchvision.utils.save_image(x_sample.cpu().detach(),
                                         'imsaves_shepardmetzlar/imsave_epoch'+str(epoch)+'_batch'+str(batch_idx)+'.png')
        train_loss = train_loss + mean_neg_elbo.item()
    print '==> TRAINING EPOCH {} Average Loss: {:.4f}'.format(
        epoch,
        train_loss/(len(train_loader.dataset))
    )

# def test(epoch):
#     mean_test_loss = 0
#     mean_kld_loss = 0
#     mean_nll_loss = 0
#     for batch_idx, (data, _) in enumerate(test_loader):
#         data = data.view(-1,105*105)
#         data = data.to(device)
#
#         z_sample, x_mean, neg_elbo, kld, nll = model(data)
#
#         mean_test_loss += neg_elbo.item()
#         mean_kld_loss += kld
#         mean_nll_loss += nll
#     mean_test_loss /= len(test_loader.dataset)
#     mean_kld_loss /= len(test_loader.dataset)
#     mean_nll_loss /= len(test_loader.dataset)
#     print '==> TESTING EPOCH {} \t Average Loss: {:.4f} \t Average KLD {:.6f} \t Average NLL {:.6f}'.format(
#         epoch,
#         mean_test_loss,
#         mean_kld_loss,
#         mean_nll_loss
#     )

z_dim = 200
h_dim = 800
batch_size = 10
learning_rate = 1e-3
n_epochs = 1000
save_every = 10
print_every = 10
clip = 10

model = VariationalModelConvDeconvOmniglot(h_dim, z_dim).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('/home/gautam/git/practice-shepardmetzlar-dataset-generator/datasets/',
                         transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs
)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs
# )

for epoch in range(1, n_epochs + 1):
    train(epoch)
    # test(epoch)
    # if epoch%save_every == 1:
    #     filename = 'modelsaves/vae_state_dict_'+str(epoch)+'.pth'
    #     torch.save(model.state_dict(), filename)
    #     print 'Model saved to',filename