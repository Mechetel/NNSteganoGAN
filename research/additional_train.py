#!/usr/bin/env python3

import sys, os
sys.path.append('..')

import torch

from steganogan_wgan_gp.models import SteganoGAN
from steganogan_wgan_gp.critics import BasicCritic
from steganogan_wgan_gp.decoders import BasicDecoder, DenseDecoder
from steganogan_wgan_gp.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from steganogan_wgan_gp.loader import DataLoader


def main():
    torch.manual_seed(42)
    training_type = 'wgan'
    timestamp = '1752271900'

    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)

    steganogan = SteganoGAN.load(
        path=os.path.join("models", training_type, timestamp, "20.rsbpp-0.964984.p"),
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models', training_type, timestamp)
    )

    steganogan.fit(train, validation, epochs=13, start_epoch=21)
    steganogan.save(os.path.join("models", training_type, timestamp, "weights.steg"))

if __name__ == '__main__':
    main()
