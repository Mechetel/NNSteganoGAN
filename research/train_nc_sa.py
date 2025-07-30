#!/usr/bin/env python3
import sys, os
sys.path.append('..')

import json
from time import time

import torch

from steganogan_nc_sa.models import SteganoGAN
from steganogan_nc_sa.decoders import DenseDecoder
from steganogan_nc_sa.encoders import DenseEncoderWithAttention
from steganogan_nc_sa.loader import DataLoader


def main():
    torch.manual_seed(42)
    training_type = 'nc_sa_dense'
    timestamp = str(int(time()))

    train = DataLoader(os.path.join("data", "div2k", "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", "div2k", "val"), shuffle=False)

    steganogan = SteganoGAN(
        data_depth=1,
        encoder=DenseEncoderWithAttention,
        decoder=DenseDecoder,
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models', training_type, timestamp)
    )
    with open(os.path.join("models", training_type, timestamp, "config.json"), "wt") as fout:
        fout.write(json.dumps({
            "epochs": 32,
            "encoder": "dense_with_attention",
            "data_depth": 1,
            "dataset": "div2k",
        }, indent=2, default=lambda o: str(o)))

    steganogan.fit(train, validation, epochs=32)
    steganogan.save(os.path.join("models", training_type, timestamp, "weights.steg"))

if __name__ == '__main__':
    main()
