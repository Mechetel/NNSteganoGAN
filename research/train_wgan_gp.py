#!/usr/bin/env python3
import sys, os
sys.path.append('..')

import json
from time import time

import torch

from steganogan_wgan_gp.models import SteganoGAN
from steganogan_wgan_gp.critics import BasicCritic
from steganogan_wgan_gp.decoders import BasicDecoder, DenseDecoder
from steganogan_wgan_gp.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from steganogan_wgan_gp.loader import DataLoader


def main():
    torch.manual_seed(42)
    training_type = 'wgan_gp'
    timestamp = str(int(time()))

    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)

    steganogan = SteganoGAN(
        data_depth=1,
        encoder=DenseEncoder,
        decoder=DenseDecoder,
        critic=BasicCritic,
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models', training_type, timestamp)
    )
    with open(os.path.join("models", training_type, timestamp, "config.json"), "wt") as fout:
        fout.write(json.dumps({
            "epochs": 32,
            "encoder": "dense",
            "data_depth": 1,
            "dataset": "div2k",
        }, indent=2, default=lambda o: str(o)))

    steganogan.fit(train, validation, epochs=32)
    steganogan.save(os.path.join("models", training_type, timestamp, "weights.steg"))

if __name__ == '__main__':
    main()
