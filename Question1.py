import random
import numpy as np
from sympy import mod_inverse
import time

# Affine Cipher Implementation
class AffineCipher:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.m = 26  # Alphabet size
        if np.gcd(a, self.m) != 1:
            raise ValueError("'a' must be coprime with 26")
        self.a_inv = mod_inverse(a, self.m)

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext.upper():
            if char.isalpha():
                x = ord(char) - ord('A')
                y = (self.a * x + self.b) % self.m
                ciphertext += chr(y + ord('A'))
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext:
            if char.isalpha():
                y = ord(char) - ord('A')
                x = (self.a_inv * (y - self.b)) % self.m
                plaintext += chr(x + ord('A'))
            else:
                plaintext += char
        return plaintext

# Columnar Transposition Cipher Implementation
def columnar_transposition_encrypt(text, key):
    num_columns = len(key)
    num_rows = -(-len(text) // num_columns)
    
    grid = [[' ' for _ in range(num_columns)] for _ in range(num_rows)]
    index = 0
    for row in range(num_rows):
        for col in range(num_columns):
            if index < len(text):
                grid[row][col] = text[index]
                index += 1
    
    sorted_key_indices = sorted(range(len(key)), key=lambda x: key[x])
    encrypted_text = "".join("".join(row[i] for row in grid) for i in sorted_key_indices)
    return encrypted_text

def columnar_transposition_decrypt(text, key):
    num_columns = len(key)
    num_rows = -(-len(text) // num_columns)
    
    grid = [[' ' for _ in range(num_columns)] for _ in range(num_rows)]
    
    sorted_key_indices = sorted(range(len(key)), key=lambda x: key[x])
    index = 0
    for col in sorted_key_indices:
        for row in range(num_rows):
            if index < len(text):
                grid[row][col] = text[index]
                index += 1
    
    decrypted_text = "".join("".join(row) for row in grid).strip()
    return decrypted_text

# Combined Encryption (Affine + Double Transposition)
def encrypt_combined(plaintext, affine_a, affine_b, key1, key2):
    affine_cipher = AffineCipher(affine_a, affine_b)
    step1 = affine_cipher.encrypt(plaintext)
    step2 = columnar_transposition_encrypt(step1, key1)
    step3 = columnar_transposition_encrypt(step2, key2)
    return step3

# Combined Decryption (Double Transposition + Affine)
def decrypt_combined(ciphertext, affine_a, affine_b, key1, key2):
    affine_cipher = AffineCipher(affine_a, affine_b)
    step1 = columnar_transposition_decrypt(ciphertext, key2)
    step2 = columnar_transposition_decrypt(step1, key1)
    step3 = affine_cipher.decrypt(step2)
    return step3

# Example Run
if __name__ == "__main__":
    plaintext = "HELLOWORLD"
    affine_a, affine_b = 5, 8  # Must choose 'a' such that gcd(a, 26) = 1
    key1 = [3, 1, 4, 2, 0]  # Columnar Transposition Key
    key2 = [2, 0, 3, 1]  # Second Columnar Transposition Key
    
      # Measure encryption time
    start_time = time.time()
    encrypted = encrypt_combined(plaintext, affine_a, affine_b, key1, key2)
    encryption_time = time.time() - start_time
    
    # Measure decryption time
    start_time = time.time()
    decrypted = decrypt_combined(encrypted, affine_a, affine_b, key1, key2)
    decryption_time = time.time() - start_time
    
    print(f"Original: {plaintext[:50]}...")  # Print a portion if too long
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted[:50]}...")
    print(f"Encryption Time: {encryption_time:.6f} seconds")
    print(f"Decryption Time: {decryption_time:.6f} seconds")