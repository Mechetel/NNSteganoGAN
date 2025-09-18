#!/usr/bin/env python3

import sys, os
sys.path.append('..')

import torch

from steganogan_nc_aes.models import SteganoGAN
from steganogan_nc_aes.decoders import BasicDecoder, DenseDecoder
from steganogan_nc_aes.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from steganogan_nc_aes.loader import DataLoader


def main():
    torch.manual_seed(42)
    training_type = 'no_critic_with_aes'
    timestamp = '1758232562'

    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)

    steganogan = SteganoGAN.load(
        path=os.path.join("models", training_type, timestamp, "32.rsbpp-0.708855.p"),
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models', training_type, timestamp)
    )

    steganogan.fit(train, validation, epochs=32, start_epoch=33)
    steganogan.save(os.path.join("models", training_type, timestamp, "weights.steg"))

if __name__ == '__main__':
    main()
