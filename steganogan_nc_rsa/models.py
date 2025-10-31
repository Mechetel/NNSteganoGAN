# -*- coding: utf-8 -*-
import gc
import inspect
import json
import os
from collections import Counter

import imageio
from PIL import Image

import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganogan_nc_rsa.utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits, most_common_candidate_from
from steganogan_nc_rsa.rsa_cypher import RSACipher

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.ssim',
    'val.psnr',
    'val.rsbpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
]


class SteganoGAN(object):
    def _get_instance(self, class_or_instance, kwargs):
        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)


    def set_device(self, cuda=True):
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device')
            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)


    def __init__(self, data_depth, encoder, decoder, cuda=False, verbose=False, log_dir=None, 
                 rsa_public_key_path=None, rsa_private_key_path=None, **kwargs):
        self.verbose = verbose

        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth

        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)

        self.set_device(cuda)

        self.encoder_decoder_optimizer = None

        # RSA cipher setup
        self.rsa_cipher = RSACipher(key_size=2048)
        
        # Load keys if provided
        if rsa_public_key_path and os.path.exists(rsa_public_key_path):
            self.rsa_cipher.load_public_key_from_file(rsa_public_key_path)
            if self.verbose:
                print(f'Loaded RSA public key from {rsa_public_key_path}')
        
        if rsa_private_key_path and os.path.exists(rsa_private_key_path):
            self.rsa_cipher.load_keys_from_files(rsa_private_key_path)
            if self.verbose:
                print(f'Loaded RSA private key from {rsa_private_key_path}')
        
        # If no keys loaded, generate new ones
        if not self.rsa_cipher.public_key:
            self.rsa_cipher.generate_keys()
            if self.verbose:
                print('Generated new RSA key pair')

        # Misc
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)


    def _random_data(self, cover):
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)


    def _encode_decode(self, cover, quantize=False):
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)

        return generated, payload, decoded


    def _get_optimizers(self):
        _enc_dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        encoder_decoder_optimizer = Adam(_enc_dec_list, lr=1e-4)

        return encoder_decoder_optimizer


    def _fit_coders(self, train, metrics):
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)

            self.encoder_decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss).backward()
            self.encoder_decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())


    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc


    def _validate(self, validate, metrics):
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.rsbpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))


    def _generate_samples(self, samples_path, epoch, text_to_encode):
        callback_images_path = os.path.join('data', 'callback_images')
        image_filenames = sorted(os.listdir(callback_images_path))
        if len(image_filenames) < 8:
            raise ValueError("Expected at least 8 generated images in callback_images")

        reshaped_tensors = []
        original_images = []

        for filename in image_filenames[:8]:
            path = os.path.join(callback_images_path, filename)

            # Load and convert to tensor (-1.0..1.0)
            image = imread(path, pilmode='RGB') / 127.5 - 1.0
            tensor = torch.FloatTensor(image).permute(2, 0, 1)

            # Apply encoding if text is provided
            if text_to_encode:
                # Prepare image for encoding (add batch dimension)
                cover = tensor.unsqueeze(0).to(self.device)
                cover_size = cover.size()

                payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text_to_encode)
                payload = payload.to(self.device)

                # Encode the image
                encoded_tensor = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)
                # Remove batch dimension and move to CPU
                tensor = encoded_tensor.squeeze(0).cpu()

            # Save original image (without resize)
            original_tensor = tensor.clamp(-1.0, 1.0)
            original_tensor = ((original_tensor + 1.0) / 2.0 * 255.0).byte()
            original_image = Image.fromarray(original_tensor.permute(1, 2, 0).numpy())
            original_images.append((filename, original_image))

            # Scale for grid
            resized_tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(360, 360), mode='bilinear', align_corners=False
            ).squeeze(0)
            reshaped_tensors.append(resized_tensor)

        # Create grid
        batch = torch.stack(reshaped_tensors).clamp(-1.0, 1.0)
        batch = ((batch + 1.0) / 2.0 * 255.0).byte()
        images = [Image.fromarray(t.permute(1, 2, 0).numpy()) for t in batch]

        grid_cols = 4
        grid_rows = 2
        gap = 20
        img_w, img_h = images[0].size
        total_w = grid_cols * img_w + (grid_cols - 1) * gap
        total_h = grid_rows * img_h + (grid_rows - 1) * gap
        grid_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))

        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * (img_w + gap)
            y = row * (img_h + gap)
            grid_img.paste(img, (x, y))

        # Save grid
        grid_filename = f'grid_epoch_{epoch}.png'
        grid_output_path = os.path.join(samples_path, grid_filename)
        grid_img.save(grid_output_path)


    def fit(self, train, validate, epochs=32, start_epoch=1, data_depth=None):
        if self.data_depth is None:
            self.data_depth = data_depth

        if self.encoder_decoder_optimizer is None:
            self.encoder_decoder_optimizer = self._get_optimizers()

        # Load existing history if metrics.log exists
        metrics_path = os.path.join(self.log_dir, 'metrics.log')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as metrics_file:
                    self.history = json.load(metrics_file)
                    print(self.history)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing metrics.log: {e}")
                self.history = []
        else:
            self.history = []

        # Start training
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(epoch, end_epoch - 1))

            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                # first epoch, every 5 epochs, and last epoch
                if epoch == start_epoch or epoch % 5 == 0 or epoch == end_epoch - 1:
                    save_name = '{}.rsbpp-{:03f}.p'.format(epoch, self.fit_metrics['val.rsbpp'])
                    self.save(os.path.join(self.log_dir, save_name))
                    self._generate_samples(self.samples_path, epoch, text_to_encode="Hello, SteganoGAN!")

            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()


    def _make_payload(self, width, height, depth, text, encrypt: bool = True):
        """
        Build payload bits for embedding.
        
        If encrypt=True, the text will be encrypted with RSA using the public key.
        The encrypted base64 string will be encoded to bits.
        """
        if text is None:
            text = ''

        if encrypt and self.rsa_cipher.public_key:
            # Encrypt text using RSA, result is base64 string
            try:
                text_to_store = self.rsa_cipher.encrypt(text)
            except Exception as e:
                raise ValueError(f"Failed to encrypt payload: {e}")
        else:
            text_to_store = text

        # Use existing pipeline to convert text to bits
        message = text_to_bits(text_to_store) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)


    def encode(self, cover, output, text):
        """
        Encode text into cover image using RSA encryption (no password needed).
        The text is encrypted with the public key automatically.
        """
        cover = imread(cover, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        # Payload is encrypted with RSA using public key
        payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text, encrypt=True)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')


    def decode(self, image):
        """
        Decode text from image using RSA decryption (no password needed).
        The private key is used automatically for decryption.
        """
        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        # Extract a bit vector
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        decoded = self.decoder(image).view(-1) > 0

        # Split and decode messages
        bits = decoded.data.int().cpu().numpy().tolist()
        decoded_data_candidate = most_common_candidate_from(decoded)

        if not decoded_data_candidate:
            raise ValueError('Failed to find message.')

        # Attempt RSA decryption of the decoded data
        if not self.rsa_cipher.private_key:
            raise ValueError('Private key is required for decryption!')

        try:
            decrypted = self.rsa_cipher.decrypt(decoded_data_candidate)
        except Exception as e:
            raise ValueError(f'Failed to decrypt message: {e}')

        return decrypted


    def save(self, path):
        torch.save(self, path)


    @classmethod
    def load(cls, path, cuda=True, verbose=False, log_dir=None, 
             rsa_public_key_path=None, rsa_private_key_path=None):
        if path is None:
            raise ValueError('Please provide a path to pretrained model.')

        if cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        steganogan = torch.load(path, map_location=device, weights_only=False)
        steganogan.verbose = verbose

        steganogan.encoder_decoder_optimizer = None

        steganogan.fit_metrics = None
        steganogan.history = list()

        steganogan.log_dir = log_dir
        if log_dir:
            os.makedirs(steganogan.log_dir, exist_ok=True)
            steganogan.samples_path = os.path.join(steganogan.log_dir, 'samples')
            os.makedirs(steganogan.samples_path, exist_ok=True)

        # Load RSA keys if provided
        if rsa_public_key_path and os.path.exists(rsa_public_key_path):
            steganogan.rsa_cipher.load_public_key_from_file(rsa_public_key_path)
            if verbose:
                print(f'Loaded RSA public key from {rsa_public_key_path}')
        
        if rsa_private_key_path and os.path.exists(rsa_private_key_path):
            steganogan.rsa_cipher.load_keys_from_files(rsa_private_key_path)
            if verbose:
                print(f'Loaded RSA private key from {rsa_private_key_path}')

        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()

        steganogan.set_device(cuda)
        return steganogan