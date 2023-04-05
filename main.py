import torch
import os
import argparse
import datetime

from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer, VQGanVAETrainerEMA, MaskGit, MaskGitTransformer
from data.fastmridata import FastMRIDataset
from data.ixidata import IXIDataset

def get_args_parser():
    parser = argparse.ArgumentParser('Maskgit MR Reconstruction')
    parser.add_argument('--model', default='vqgan', type=str, choices=['vqgan', 'vqgan-ema', 'maskgit'])
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--output_path', default='', type=str)
    parser.add_argument('--batch_size', default=2, type=int)

    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--dataset', default='fastmri', type=str, choices=['fastmri', 'ixi'])
    parser.add_argument('--input_size', default=320, type=int)
    parser.add_argument('--domain', default='img', type=str, choices=['img', 'kspace'])
    parser.add_argument('--down', default='random', choices=['uniform', 'random'], help='method of constructing undersampled data')
    parser.add_argument('--low_freq_ratio', default=0.7, help='ratio of low frequency lines')
    parser.add_argument('--downsample', default=16, type=int, help='maximum acceleration factor')
    
    return parser


def run_vqgan(args, dataset):
    vae = VQGanVAE(
        channels=1,
        layers=4,
        dim=256,
        vq_codebook_dim=32, #256
        vq_codebook_size=2048
    )

    trainer = VQGanVAETrainer(
        vae=vae,
        dataset=dataset,
        image_size=args.input_size,
        batch_size=args.batch_size,
        grad_accum_every=8,
        num_train_steps=50000, 
        results_folder=args.output_path,
        use_ema=True
    ).cuda()
    trainer.train()



def run_vqgan_ema(args, dataset):
    vae = VQGanVAE(
        channels=1,
        layers=4,
        dim=256,
        vq_codebook_dim=32, #256
        vq_codebook_size=2048
    )

    trainer = VQGanVAETrainerEMA(
        vae=vae,
        dataset=dataset,
        image_size=args.input_size,
        batch_size=args.batch_size,
        grad_accum_every=8,
        num_train_steps=50000,
        results_folder=args.output_path,
        use_ema=True
    ).cuda()
    trainer.train()

    return 


def run_maskgit(args, dataset):

    # first instantiate your vae

    vae = VQGanVAE(
        dim = 256,
        vq_codebook_size = 512
    ).cuda()

    vae.load('/path/to/vae.pt') # you will want to load the exponentially moving averaged VAE

    # then you plug the vae and transformer into your MaskGit as so

    # (1) create your transformer / attention network

    transformer = MaskGitTransformer(
        num_tokens = 512,         # must be same as codebook size above
        seq_len = 256,            # must be equivalent to fmap_size ** 2 in vae
        dim = 512,                # model dimension
        depth = 8,                # depth
        dim_head = 64,            # attention head dimension
        heads = 8,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        t5_name = 't5-small',     # name of your T5
    )

    # (2) pass your trained VAE and the base transformer to MaskGit

    base_maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = 256,          # image size
        cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    ).cuda()

    # ready your training text and images

    texts = [
        'a child screaming at finding a worm within a half-eaten apple',
        'lizard running across the desert on two feet',
        'waking up to a psychedelic landscape',
        'seashells sparkling in the shallow waters'
    ]

    images = torch.randn(4, 3, 256, 256).cuda()

    # feed it into your maskgit instance, with return_loss set to True

    loss = base_maskgit(
        images,
        texts = texts
    )

    loss.backward()

    # do this for a long time on much data
    # then...

    images = base_maskgit.generate(texts = [
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles'
    ], cond_scale = 3.) # conditioning scale for classifier free guidance

    images.shape # (3, 3, 256, 256)
    return



if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset=='fastmri':
        dataset=FastMRIDataset(args, mode=args.mode)
    elif args.dataset=='ixi':
        dataset=IXIDataset(args, mode=args.mode)
    
    dt = datetime.datetime.now()
    base = '{}_{}'.format(dt.strftime('%m%d'), args.model)
    args.output_path = os.path.join(args.data_path, args.dataset, 'checkpoints', base)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if args.model=='vqgan':
        run_vqgan(args, dataset)
    elif args.model=='vqgan-ema':
        run_vqgan_ema(args, dataset)
    elif args.model=='maskgit':
        run_maskgit(args, dataset)


