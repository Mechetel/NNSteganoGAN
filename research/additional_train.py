#!/usr/bin/env python3

import json
import os

import torch

import sys, os
sys.path.append('..')

from steganogan.models import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from steganogan.loader import DataLoader


def main():
    torch.manual_seed(42)
    training_type = 'original'
    timestamp = '1751999939'

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
