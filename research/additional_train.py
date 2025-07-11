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
    timestamp = '1751999939'

    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)

    steganogan = SteganoGAN.load(
        path=os.path.join("models", timestamp, "weights.steg"),
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models1', timestamp)
    )

    with open(os.path.join("models1", timestamp, "config.json"), "wt") as fout:
        fout.write(
            json.dumps({
                "epochs": 64,
                "encoder": "dense",
                "data_depth": 1,
                "dataset": "div2k",
            }, indent=2, default=lambda o: str(o))
        )

    steganogan.fit(train, validation, epochs=32, start_epoch=33, data_depth=1)
    steganogan.save(os.path.join("models1", timestamp, "weights.steg"))

if __name__ == '__main__':
    main()
