import sys, os
sys.path.append('..')

from steganogan.models import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from steganogan.loader import DataLoader


DATA_DEPTH = 1
LOG_DIR = 'logs'
MODEL_PATH = f'models/dense_{DATA_DEPTH}.steg'


train = DataLoader("../research/data/div2k/train", shuffle=True)
validation = DataLoader("../research/data/div2k/val", shuffle=False)

steganogan = SteganoGAN(DATA_DEPTH, DenseEncoder, DenseDecoder, BasicCritic, cuda=False, verbose=True, log_dir=LOG_DIR)
# steganogan = SteganoGAN.load(architecture=None, path=MODEL_PATH, cuda=False, verbose=True)
steganogan.fit(validation, validation, epochs=32, start_epoch=1)
steganogan.save(MODEL_PATH)


steganogan = SteganoGAN.load(architecture=None, path=MODEL_PATH, cuda=False, verbose=True)
steganogan.encode('images/testing/input1.png', 'images/testing_output/output1.png', 'Hi')
steganogan.decode('images/testing_output/output1.png')

steganogan.encode('images/testing/input2.png', 'images/testing_output/output2.png', 'Hello World')
steganogan.decode('images/testing_output/output2.png')

steganogan.encode('images/testing/input3.png', 'images/testing_output/output3.png', 'This is a test message')
steganogan.decode('images/testing_output/output3.png')

steganogan.encode('images/testing/input4.png', 'images/testing_output/output4.png', 'SteganoGAN is awesome!')
steganogan.decode('images/testing_output/output4.png')
