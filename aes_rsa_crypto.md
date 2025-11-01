=============================================================================
STEGANOGAN + AES
=============================================================================

ENCODING:
---------
input: cover_image, message, password

1. salt ← random(16)
2. key ← pbkdf2(password, salt, 100k_iterations)
3. iv ← random(16)
4. ciphertext ← aes_cbc_encrypt(message, key, iv)
5. password_bits ← to_bits(password)
6. data_bits ← to_bits(salt + iv + ciphertext)
7. stego ← gan_encode(cover_image, password_bits + data_bits)
8. *return* stego

DECODING:
---------
input: stego_image, password

1. bits ← gan_decode(stego_image)
2. password_bits, data_bits ← split(bits)
3. extracted_password ← from_bits(password_bits)
4. *if* extracted_password ≠ password: *raise* error
5. salt, iv, ciphertext ← parse(from_bits(data_bits))
6. key ← pbkdf2(password, salt, 100k_iterations)
7. message ← aes_cbc_decrypt(ciphertext, key, iv)
8. *return* message


=============================================================================
STEGANOGAN + RSA
=============================================================================

KEY GENERATION:
---------------
1. (public_key, private_key) ← generate_rsa_keys(2048)
2. save(public_key, private_key)

ENCODING:
---------
input: cover_image, message, public_key

1. ciphertext ← rsa_oaep_encrypt(message, public_key)
2. bits ← to_bits(ciphertext)
3. stego ← gan_encode(cover_image, bits)
4. *return* stego

DECODING:
---------
input: stego_image, private_key

1. bits ← gan_decode(stego_image)
2. ciphertext ← from_bits(bits)
3. message ← rsa_oaep_decrypt(ciphertext, private_key)
4. *return* message


=============================================================================
STEGANOGAN + NEURAL CRYPTO
=============================================================================

ENCODING (SIMPLE):
------------------
input: cover_image, message, password

1. embedding ← hash_password_multiple(password, 512)
2. neural_key ← neural_network(embedding)
3. payload ← [0x00, 0, message]
4. iv ← random(16)
5. encrypted ← aes_gcm(payload, neural_key, iv)
6. final ← [version, flags, checksum] + iv + tag + encrypted
7. bits ← to_bits(base64(final))
8. stego ← gan_encode(cover_image, bits)
9. *return* stego

ENCODING (IMAGE-LOCK):
----------------------
input: cover_image, message, password

1. phash ← perceptual_hash(cover_image)
2. embedding ← hash_password_multiple(password, 512)
3. neural_key_temp ← neural_network(embedding)
4. final_key ← hash(neural_key_temp + phash)
5. payload ← [0x01, 8, phash, message]
6. iv ← random(16)
7. encrypted ← aes_gcm(payload, final_key, iv)
8. final ← [version, flags, checksum] + iv + tag + encrypted
9. bits ← to_bits(base64(final))
10. stego ← gan_encode(cover_image, bits)
11. *return* stego

DECODING (SIMPLE):
------------------
input: stego_image, password

1. bits ← gan_decode(stego_image)
2. data ← base64_decode(from_bits(bits))
3. version, flags, checksum, iv, tag, encrypted ← parse(data)
4. embedding ← hash_password_multiple(password, 512)
5. neural_key ← neural_network(embedding)
6. *if* not verify(tag): *raise* error
7. payload ← aes_gcm_decrypt(encrypted, neural_key, iv)
8. flags, hash_len, message ← parse(payload)
9. *return* message

DECODING (IMAGE-LOCK):
----------------------
input: stego_image, password, cover_image

1. bits ← gan_decode(stego_image)
2. data ← base64_decode(from_bits(bits))
3. version, flags, checksum, iv, tag, encrypted ← parse(data)
4. phash ← perceptual_hash(cover_image)
5. embedding ← hash_password_multiple(password, 512)
6. neural_key_temp ← neural_network(embedding)
7. final_key ← hash(neural_key_temp + phash)
8. *if* not verify(tag): *raise* error
9. payload ← aes_gcm_decrypt(encrypted, final_key, iv)
10. flags, hash_len, embedded_phash, message ← parse(payload)
11. *if* embedded_phash ≠ phash: *raise* error
12. *return* message


=============================================================================
SUMMARY
=============================================================================

aes: pbkdf2 → aes_cbc → embed password + data
rsa: rsa_oaep → embed data only
neural: neural_net → aes_gcm → optional image_hash → self-contained