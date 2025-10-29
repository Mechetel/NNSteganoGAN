# AES helper (uses your implementation)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.padding import PKCS7
import base64
import os

class AESCipher:
    def __init__(self, password: str):
        self.password = password.encode('utf-8')

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self.password)

    def encrypt(self, plaintext: str) -> str:
        salt = os.urandom(16)
        iv = os.urandom(16)
        key = self._derive_key(salt)

        data = plaintext.encode('utf-8')
        padder = PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        encrypted_data = salt + iv + ciphertext
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, encrypted_text: str) -> str:
        try:
            encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
            salt = encrypted_data[:16]
            iv = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            key = self._derive_key(salt)

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()

            unpadder = PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            return data.decode('utf-8')
        except Exception as e:
            # keep returning False-like behavior consistent with existing bytearray_to_text
            raise ValueError(f"Ошибка расшифровки: {str(e)}. Возможно, неверный пароль.")
