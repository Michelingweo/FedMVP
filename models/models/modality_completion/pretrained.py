import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter
import os
from torchvision.utils import save_image
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

from utils.dataset_utils import get_cub_200_2011, get_oxford_flowers_102
from utils.options import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
# openai pretrained clip - defaults to ViT-B/32
clip = OpenAIClipAdapter()

#data
epoch_dp = 5
epoch_dec = 5

# load dataset and split users
if args.dataset == 'cub': # total 11788 images
    # 8855 train images
    train_set, train_loader = get_cub_200_2011(split='train_val', d_batch=64)
    # 2933 test images
    # test_set, test_loader = get_cub_200_2011(split='test', d_batch=64)

elif args.dataset == 'flower':

    train_set, train_loader = get_oxford_flowers_102(split='train_val', d_batch=64)
    test_set, test_loader = get_oxford_flowers_102(split='test', d_batch=64)
else:
    exit('Error: unrecognized dataset')

img_size = train_set[0][0].shape



#
# text = torch.randint(0, 49408, (4, 256)).cuda()
# images = torch.randn(4, 3, 256, 256).cuda()


# prior networks (with transformer)

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).to(args.device)

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
).to(args.device)

# diffusion prior training
optimizer_dp = optim.Adam(diffusion_prior.parameters(), lr=args.lr)
diffusion_prior.train()

# list_loss = []
for i in range(epoch_dp):
    # batch_loss = []
    for batch_idx,(images, idx, text) in enumerate(train_loader):
        images = images.cuda()
        print(text)
        print(torch.tensor(text[0]))
        text = [torch.tensor(_) for _ in text]
        # text = torch.tensor(text).cuda()
        text = pad_sequence(text, batch_first=True).cuda()

        idx = idx.cuda()
        optimizer_dp.zero_grad()
        loss_dp = diffusion_prior(text, images).to(args.device)
        loss_dp.backward()
        optimizer_dp.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_dp.item()))

# do above for many steps ...

# decoder (with unet)

unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    text_embed_dim = 512,
    cond_on_text_encodings = True  # set to True for any unets that need to be conditioned on text encodings (ex. first unet in cascade)
).to(args.device)

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).to(args.device)

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 1000,
    sample_timesteps = (250, 27),
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).to(args.device)

# decoder training
list_loss = []
decoder.train()
optimizer_dec = optim.Adam(decoder.state_dict(), lr=args.lr)

for j in range(epoch_dec):
    batch_loss = []
    for batch_idx, (images, idx, text) in enumerate(train_loader):
        images = images.cuda()

        text = [torch.tensor(_).cuda() for _ in text]
        # text = torch.tensor(text).cuda()
        idx = idx.cuda()
        for unet_number in (1, 2):
            optimizer_dec.zero_grad()
            loss_dec = decoder(images, text = text, unet_number = unet_number).to(args.device) # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
            loss_dec.backward()
            optimizer_dec.step()


        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                j, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_dec.item()))
        batch_loss.append(loss_dec.item())
    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain loss:', loss_avg)
    list_loss.append(loss_avg)

# plot loss
plt.figure()
plt.plot(range(len(list_loss)), list_loss)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.savefig('./log/decoder_{}_{}.png'.format(args.dataset, epoch_dec))



#model save
save_path = '../../save/trained_model/'
prior_network_path = os.path.join(save_path,'prior_network_{}_{}.pth'.format(args.dataset, epoch_dp))
diffusion_prior_path = os.path.join(save_path,'diffusion_prior_{}_{}.pth'.format(args.dataset, epoch_dp))
unet1_path = os.path.join(save_path,'unet1_{}_{}.pth'.format(args.dataset, epoch_dec))
unet2_path = os.path.join(save_path,'unet2_{}_{}.pth'.format(args.dataset, epoch_dec))
decoder_path = os.path.join(save_path,'decoder_{}_{}.pth'.format(args.dataset, epoch_dec))



torch.save({
            'epoch': epoch_dp,
            'model_state_dict': prior_network.state_dict(),
            'optimizer_state_dict': optimizer_dp.state_dict(),
            'loss': loss_dp
            }, prior_network_path)
torch.save({
            'epoch': epoch_dp,
            'model_state_dict': diffusion_prior .state_dict(),
            'optimizer_state_dict': optimizer_dp.state_dict(),
            'loss': loss_dp
            }, diffusion_prior_path)
torch.save({
            'epoch': epoch_dec,
            'model_state_dict': unet1.state_dict(),
            'optimizer_state_dict': optimizer_dec.state_dict(),
            'loss': loss_dec
            }, unet1_path)
torch.save({
            'epoch': epoch_dec,
            'model_state_dict': unet2.state_dict(),
            'optimizer_state_dict': optimizer_dec.state_dict(),
            'loss': loss_dec
            }, unet2_path)
torch.save({
            'epoch': epoch_dec,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer_dec.state_dict(),
            'loss': loss_dec
            }, decoder_path)






# do above for many steps
