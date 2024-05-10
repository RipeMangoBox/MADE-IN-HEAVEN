import numpy as np
import torch
import torch.nn.functional as F
import json
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, GatedPixelCNN

from pixelcnn_baseline import generate_samples


def generate():
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE with PixelCNN prior for Generation')

    # General
    parser.add_argument('--vqvae', type=str,default='models/vqvae/best.pt',
        help='filename containing the vqvae model')
    parser.add_argument('--prior', type=str,default='models/pixelcnn_prior/prior.pt',
        help='filename containing the prior model')
    parser.add_argument('--dataset', type=str,default='cifar10',
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet) (default=cifar10)')
    parser.add_argument('--num_channels', type=int, default='3',
        help='channels of the dataset (default: 3)')
    
    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')
    
    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='./samples/samples_generated',
        help='name of the output folder (default: ./samples/samples_generated)')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cuda)')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('{0}'.format(args.output_folder)):
        os.makedirs('{0}'.format(args.output_folder))

    # lodel vqvae
    vqvae = VectorQuantizedVAE(args.num_channels, args.hidden_size_vae, args.k).to(args.device)
    with open(args.vqvae, 'rb') as f:
        state_dict = torch.load(f)
        vqvae.load_state_dict(state_dict)
    vqvae.eval()

    prior = GatedPixelCNN(args.k, args.hidden_size_prior, args.num_layers, n_classes=50000).to(args.device)
    with open(args.prior, 'rb') as f:
        state_dict = torch.load(f)
        prior.load_state_dict(state_dict)
    prior.eval()


    with torch.no_grad():
        labels = torch.arange(10).expand(10, 10).contiguous().view(-1).to(args.device)
        generate_samples(labels, 'cifar10')


if __name__=='__main__':
    generate()