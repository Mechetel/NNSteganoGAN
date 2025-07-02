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

from steganogan.utils import ssim, text_to_bits, bits_to_text

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.rsbpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


class SteganoGAN(object):

    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class"""

        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        """Sets the torch device depending on whether cuda is avaiable or not."""
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
        self.critic.to(self.device)

    def __init__(self, data_depth, encoder, decoder, critic, cuda=False, verbose=False, log_dir=None, **kwargs):
        self.verbose = verbose
        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth

        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)

        self.critic_optimizer = None
        self.encoder_decoder_optimizer = None

        # Misc
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.callback_images_path = os.path.join(self.log_dir, 'callback_images')
            self.epoch_images_path = os.path.join(self.log_dir, 'epoch_images')
            self.grid_epoch_images_path = os.path.join(self.log_dir, 'grid_epoch_images')
            os.makedirs(self.callback_images_path, exist_ok=True)
            os.makedirs(self.epoch_images_path, exist_ok=True)
            os.makedirs(self.grid_epoch_images_path, exist_ok=True)


    def _random_data(self, cover):
        """Generate random data ready to be hidden inside the cover image.

        Args:
            cover (image): Image to use as cover.

        Returns:
            generated (image): Image generated with the encoded message.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _encode_decode(self, cover, quantize=False):
        """Encode random data and then decode it.

        Args:
            cover (image): Image to use as cover.
            quantize (bool): whether to quantize the generated image or not.

        Returns:
            generated (image): Image generated with the encoded message.
            payload (bytes): Random data that has been encoded in the image.
            decoded (bytes): Data decoded from the generated image.
        """
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)

        return generated, payload, decoded

    def _critic(self, image):
        """Evaluate the image using the critic"""
        return torch.mean(self.critic(image))

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        encoder_decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return critic_optimizer, encoder_decoder_optimizer

    def _fit_critic(self, train, metrics):
        """Critic process"""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward()
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

    def _fit_coders(self, train, metrics):
        """Fit the encoder and the decoder on the train images."""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)
            generated_score = self._critic(generated)

            self.encoder_decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss + generated_score).backward()
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
        """Validation process"""
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)
            generated_score = self._critic(generated)
            cover_score = self._critic(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.rsbpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def _generate_samples(self, epoch, text_to_encode=""):
        """
        Generate samples with optional encoding step.
        
        Args:
            epoch: Current epoch number
            text_to_encode: Text message to encode into images (empty string for no encoding)
        """
        image_filenames = sorted(os.listdir(self.callback_images_path))
        if len(image_filenames) < 8:
            raise ValueError("Expected at least 8 generated images in callback_images")
        
        reshaped_tensors = []
        original_images = []
        
        for filename in image_filenames[:8]:
            path = os.path.join(self.callback_images_path, filename)
            
            # Load and convert to tensor (-1.0..1.0)
            image = imread(path, pilmode='RGB') / 127.5 - 1.0
            tensor = torch.FloatTensor(image).permute(2, 0, 1)
            
            # Apply encoding if text is provided
            if text_to_encode:
                # Prepare image for encoding (add batch dimension)
                cover = tensor.unsqueeze(0).to(self.device)
                cover_size = cover.size()
                
                # Create payload for encoding
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
        
        # Save original images
        epoch_dir = os.path.join(self.epoch_images_path, f'epoch{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        for filename, img in original_images:
            # Add prefix to indicate if encoded
            if text_to_encode:
                base_name, ext = os.path.splitext(filename)
                filename = f"encoded_{base_name}{ext}"
            output_path = os.path.join(epoch_dir, filename)
            img.save(output_path)
        
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
        if text_to_encode:
            grid_filename = f'encoded_grid_epoch_{epoch}.png'
        
        grid_output_path = os.path.join(self.grid_epoch_images_path, grid_filename)
        grid_img.save(grid_output_path)
        
        if self.verbose:
            encoding_status = "with encoding" if text_to_encode else "without encoding"
            print(f'Original images saved to {self.epoch_images_path} ({encoding_status})')
            print(f'Grid image saved to {grid_output_path}')
            if text_to_encode:
                print(f'Encoded message: "{text_to_encode}"')


    def fit(self, train, validate, epochs=40, start_epoch=1):
        """Train a new model with the given ImageLoader class."""

        if self.critic_optimizer is None:
            self.critic_optimizer, self.encoder_decoder_optimizer = self._get_optimizers()
            
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
                print('Epoch {}/{}'.format(epoch, end_epoch - 1 ))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}.rsbpp-{:03f}.p'.format(epoch, self.fit_metrics['val.rsbpp'])

                self.save(os.path.join(self.log_dir, save_name))
                self._generate_samples(epoch, text_to_encode="Hello, SteganoGAN!")

            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()

    def _make_payload(self, width, height, depth, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        """
        message = text_to_bits(text) + [0] * 32
        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)

    def encode(self, cover, output, text):
        """Encode an image.
        Args:
            cover (str): Path to the image to be used as cover.
            output (str): Path where the generated image will be saved.
            text (str): Message to hide inside the image.
        """
        cover = imread(cover, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        # _, _, height, width = cover.size()
        payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')

    def decode(self, image):

        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        # extract a bit vector
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        # split and decode messages
        bits = image.data.int().cpu().numpy().tolist()
        text = bits_to_text(bits)
        return text

    def save(self, path):
        """Save the fitted model in the given path. Raises an exception if there is no model."""
        torch.save(self, path)

    @classmethod
    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        """Loads an instance of SteganoGAN for the given architecture (default pretrained models)
        or loads a pretrained model from a given path.

        Args:
            architecture(str): Name of a pretrained model to be loaded from the default models.
            path(str): Path to custom pretrained model. *Architecture must be None.
            cuda(bool): Force loaded model to use cuda (if available).
            verbose(bool): Force loaded model to use or not verbose.
        """
        if cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        if architecture and not path:
            model_name = '{}.steg'.format(architecture)
            pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained')
            path = os.path.join(pretrained_path, model_name)

        elif (architecture is None and path is None) or (architecture and path):
            raise ValueError(
                'Please provide either an architecture or a path to pretrained model.')

        steganogan = torch.load(path, map_location=device, weights_only=False)
        steganogan.verbose = verbose

        steganogan.set_device(cuda)
        return steganogan
