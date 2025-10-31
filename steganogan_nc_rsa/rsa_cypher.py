# RSA helper
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
import base64


class RSACipher:
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
    
    def generate_keys(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
    
    def export_private_key(self, password: str = None) -> str:
        if not self.private_key:
            raise ValueError("Сначала сгенерируйте ключи!")
        
        encryption = serialization.BestAvailableEncryption(password.encode('utf-8')) if password else serialization.NoEncryption()
        
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )
        return pem.decode('utf-8')
    
    def export_public_key(self) -> str:
        if not self.public_key:
            raise ValueError("Сначала сгенерируйте ключи!")
        
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def load_private_key(self, pem_data: str, password: str = None):
        password_bytes = password.encode('utf-8') if password else None
        self.private_key = load_pem_private_key(
            pem_data.encode('utf-8'),
            password=password_bytes
        )
        self.public_key = self.private_key.public_key()
    
    def load_public_key(self, pem_data: str):
        self.public_key = load_pem_public_key(pem_data.encode('utf-8'))
    
    def save_keys_to_files(self, private_key_path: str = "private_key.pem", 
                          public_key_path: str = "public_key.pem", 
                          password: str = None):
        if not self.private_key or not self.public_key:
            raise ValueError("Сначала сгенерируйте ключи!")
        
        private_pem = self.export_private_key(password)
        with open(private_key_path, 'w') as f:
            f.write(private_pem)
        
        public_pem = self.export_public_key()
        with open(public_key_path, 'w') as f:
            f.write(public_pem)
    
    def load_keys_from_files(self, private_key_path: str = "private_key.pem",
                            password: str = None):
        with open(private_key_path, 'r') as f:
            private_pem = f.read()
        self.load_private_key(private_pem, password)
    
    def load_public_key_from_file(self, public_key_path: str = "public_key.pem"):
        with open(public_key_path, 'r') as f:
            public_pem = f.read()
        self.load_public_key(public_pem)
    
    def encrypt(self, plaintext: str) -> str:
        if not self.public_key:
            raise ValueError("Сначала сгенерируйте или загрузите ключи!")
        
        data = plaintext.encode('utf-8')
        
        ciphertext = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.b64encode(ciphertext).decode('utf-8')
    
    def decrypt(self, encrypted_text: str) -> str:
        if not self.private_key:
            raise ValueError("Для расшифровки нужен приватный ключ!")
        
        try:
            ciphertext = base64.b64decode(encrypted_text.encode('utf-8'))
            
            plaintext = self.private_key.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return plaintext.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Ошибка расшифровки: {str(e)}. Возможно, неверный ключ.")

