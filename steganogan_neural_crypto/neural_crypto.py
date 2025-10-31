from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import hashlib
import os
import struct
from PIL import Image


# ==================== NEURAL CRYPTO COMPONENTS ====================

class NeuralKeyDerivation(nn.Module):
    """Нейронна мережа для виведення ключів шифрування"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 32)  # 256-bit ключ
        self.dropout = nn.Dropout(0.3)
        
        # Ініціалізація ваг
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


class PerceptualHashExtractor:
    """Екстрактор перцептивного хешу для image-locking"""
    @staticmethod
    def extract_phash(image_path, hash_size=8):
        """Витягує perceptual hash з зображення"""
        img = Image.open(image_path).convert('L')
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img).astype(np.float32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        phash_bits = ''.join(['1' if x else '0' for row in diff for x in row])
        return int(phash_bits, 2).to_bytes((len(phash_bits) + 7) // 8, 'big')


class NeuralCryptoEngine:
    """
    Двигун нейронної криптографії для SteganoGAN
    Повністю автономна система - БЕЗ зовнішніх метаданих!
    """
    
    VERSION = 1
    FLAG_IMAGE_LOCKED = 0x01
    FLAG_COMPRESSED = 0x02
    
    def __init__(self, device='cpu'):
        self.device = device
        self.neural_kdf = NeuralKeyDerivation().to(device)
        self.neural_kdf.eval()
        self.phash_extractor = PerceptualHashExtractor()
        # DON'T store backend - create it when needed
    
    @property
    def backend(self):
        """Lazy backend creation to avoid pickling issues"""
        return default_backend()
    
    def _passphrase_to_embedding(self, passphrase: str) -> torch.Tensor:
        """Конвертує пароль у вхід нейромережі"""
        embeddings = []
        for i in range(8):
            h = hashlib.sha512(f"{passphrase}_{i}".encode()).digest()
            embeddings.extend(h)
        
        tensor = torch.tensor([b / 255.0 for b in embeddings[:512]], dtype=torch.float32)
        return tensor.unsqueeze(0).to(self.device)
    
    def _derive_key(self, passphrase: str, image_hash: bytes = None) -> bytes:
        """Виводить ключ шифрування через нейромережу"""
        with torch.no_grad():
            passphrase_embedding = self._passphrase_to_embedding(passphrase)
            neural_key = self.neural_kdf(passphrase_embedding)
            neural_key_bytes = ((neural_key.cpu().numpy() + 1) * 127.5).astype(np.uint8).tobytes()
        
        if image_hash:
            # Змішування з image hash для image-locking
            combined = hashlib.sha256(neural_key_bytes + image_hash).digest()
            return combined[:32]
        
        return neural_key_bytes[:32]
    
    def _aes_encrypt(self, message: bytes, key: bytes) -> bytes:
        """AES-GCM шифрування з автентифікацією"""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(message) + encryptor.finalize()
        tag = encryptor.tag
        
        return iv + tag + ciphertext
    
    def _aes_decrypt(self, encrypted: bytes, key: bytes) -> bytes:
        """AES-GCM розшифрування з перевіркою автентичності"""
        if len(encrypted) < 32:
            raise ValueError("Зашифровані дані занадто короткі")
        
        iv = encrypted[:16]
        tag = encrypted[16:32]
        ciphertext = encrypted[32:]
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def encrypt(self, message: str, passphrase: str, cover_image_path: str = None) -> bytes:
        """
        Шифрує повідомлення для вбудовування в SteganoGAN
        
        Формат: [HEADER][ENCRYPTED_PAYLOAD]
        Header: [version:1][flags:1][checksum:4]
        Payload: [flags_copy:1][image_hash_len:1][image_hash][message]
        """
        message_bytes = message.encode('utf-8')
        
        # Визначаємо прапорці
        flags = 0
        image_hash = None
        
        if cover_image_path:
            flags |= self.FLAG_IMAGE_LOCKED
            image_hash = self.phash_extractor.extract_phash(cover_image_path)
        
        # Виводимо ключ шифрування
        encryption_key = self._derive_key(passphrase, image_hash)
        
        # Створюємо payload з вбудованими метаданими
        payload = struct.pack('B', flags)
        
        if image_hash:
            payload += struct.pack('B', len(image_hash)) + image_hash
        else:
            payload += struct.pack('B', 0)
        
        payload += message_bytes
        
        # Шифруємо весь payload
        encrypted_payload = self._aes_encrypt(payload, encryption_key)
        
        # Створюємо заголовок
        version = struct.pack('B', self.VERSION)
        flags_byte = struct.pack('B', flags)
        checksum_bytes = hashlib.sha256(encrypted_payload).digest()
        checksum = struct.pack('>I', 
            checksum_bytes[0] << 24 | 
            checksum_bytes[1] << 16 |
            checksum_bytes[2] << 8 |
            checksum_bytes[3])
        
        # Комбінуємо: header + encrypted payload
        final_data = version + flags_byte + checksum + encrypted_payload
        
        return final_data
    
    def decrypt(self, encrypted_data: bytes, passphrase: str, 
                cover_image_path: str = None) -> str:
        """
        Розшифровує повідомлення після екстракції з SteganoGAN
        Всі метадані витягуються з самих зашифрованих даних!
        """
        if len(encrypted_data) < 6:
            raise ValueError("Дані занадто короткі - пошкоджені або неправильна екстракція")
        
        # Парсимо заголовок
        version = struct.unpack('B', encrypted_data[0:1])[0]
        flags = struct.unpack('B', encrypted_data[1:2])[0]
        stored_checksum = struct.unpack('>I', encrypted_data[2:6])[0]
        encrypted_payload = encrypted_data[6:]
        
        # Перевіряємо checksum
        computed_checksum_bytes = hashlib.sha256(encrypted_payload).digest()
        computed_checksum = (
            computed_checksum_bytes[0] << 24 | 
            computed_checksum_bytes[1] << 16 |
            computed_checksum_bytes[2] << 8 |
            computed_checksum_bytes[3]
        )
        
        if stored_checksum != computed_checksum:
            print("⚠ Попередження: Невідповідність checksum - дані можуть бути пошкоджені")
        
        # Перевіряємо image-locking
        image_locked = bool(flags & self.FLAG_IMAGE_LOCKED)
        
        if image_locked and not cover_image_path:
            raise ValueError("Повідомлення IMAGE-LOCKED але cover image не надано!")
        
        if image_locked:
            image_hash = self.phash_extractor.extract_phash(cover_image_path)
        else:
            image_hash = None
        
        # Виводимо ключ розшифрування
        decryption_key = self._derive_key(passphrase, image_hash)
        
        # Розшифровуємо payload
        try:
            decrypted_payload = self._aes_decrypt(encrypted_payload, decryption_key)
        except Exception as e:
            raise ValueError(f"Помилка розшифрування - неправильний пароль або пошкоджені дані: {e}")
        
        # Перевіряємо мінімальну довжину
        if len(decrypted_payload) < 2:
            raise ValueError("Розшифрований payload занадто короткий")
        
        # Парсимо розшифрований payload
        payload_flags = struct.unpack('B', decrypted_payload[0:1])[0]
        image_hash_len = struct.unpack('B', decrypted_payload[1:2])[0]
        offset = 2
        
        if image_hash_len > 0:
            embedded_image_hash = decrypted_payload[offset:offset + image_hash_len]
            offset += image_hash_len
            
            # Перевіряємо image hash якщо image-locked
            if image_locked and embedded_image_hash != image_hash:
                raise ValueError("Невідповідність image hash - неправильне cover зображення або підроблені дані!")
        
        message_bytes = decrypted_payload[offset:]
        message = message_bytes.decode('utf-8')
        
        return message