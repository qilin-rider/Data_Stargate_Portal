import mpmath
import numpy as np
from fractions import Fraction
import zlib
import os
from pathlib import Path
import socket
import pickle
import time
from collections import defaultdict
import hashlib
import psutil
import logging
import itertools

# Colab-specific imports
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Set mpmath precision
mpmath.mp.dps = 500  # Reduced for speed

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler('dictionary_build_log.txt', mode='w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

# Cache for mpmath results
MPMATH_CACHE = {}

def get_prime_reciprocal(p, max_digits=500):
    try:
        frac = Fraction(1, p)
        decimal_str = str(frac)
        parts = decimal_str.split('.')
        decimal = parts[1] if len(parts) > 1 else "0"
        if p in (2, 5):
            return decimal
        period = p - 1
        while len(decimal) < max_digits:
            decimal += decimal
        return decimal[:max_digits]
    except Exception as e:
        logger.error(f"Error in get_prime_reciprocal(p={p}): {e}")
        return "0" * max_digits

def get_pi(max_digits=500):
    return MPMATH_CACHE.setdefault('pi', mpmath.mp.pi().__str__().replace('.', '')[:max_digits])

def get_sqrt2(max_digits=500):
    return MPMATH_CACHE.setdefault('sqrt2', mpmath.mp.sqrt(2).__str__().replace('.', '')[:max_digits])

def get_e(max_digits=500):
    return MPMATH_CACHE.setdefault('e', mpmath.mp.e().__str__().replace('.', '')[:max_digits])

def get_phi(max_digits=500):
    return MPMATH_CACHE.setdefault('phi', mpmath.mp.phi().__str__().replace('.', '')[:max_digits])

def get_logistic_0_1(max_digits=500):
    return logistic_map(0.1, 3.9, max_digits)

def sin_sequence(max_digits=500):
    return ''.join('1' if mpmath.sin(n/100) > 0 else '0' for n in range(max_digits))

def get_cf_sqrt2(max_digits=500):
    return continued_fraction(mpmath.mp.sqrt(2), max_digits)

def get_fib_7(max_digits=500):
    return fibonacci_modulo(7, max_digits)

def logistic_map(x0, r, n):
    try:
        x = x0
        result = []
        for _ in range(n):
            x = r * x * (1 - x)
            result.append('1' if x > 0.5 else '0')
        return ''.join(result)
    except Exception as e:
        logger.error(f"Error in logistic_map(x0={x0}, r={r}): {e}")
        return "0" * n

def continued_fraction(n, max_terms=500):
    try:
        seq = []
        x = n
        for _ in range(max_terms):
            a = int(x)
            seq.append(str(a % 2))
            x = 1 / (x - a) if x != a else 0
            if x == 0:
                break
        return ''.join(seq)
    except Exception as e:
        logger.error(f"Error in continued_fraction(n={n}): {e}")
        return "0" * max_terms

def fibonacci_modulo(p, n):
    try:
        fib = [0, 1]
        for i in range(2, n):
            fib.append((fib[i-1] + fib[i-2]) % p)
        return ''.join(str(x % 2) for x in fib[:n])
    except Exception as e:
        logger.error(f"Error in fibonacci_modulo(p={p}): {e}")
        return "0" * n

def build_dictionary(max_digits=500, chunk_sizes=[16, 32, 64, 128, 256, 512], timeout=60):
    dictionary_file = 'dictionary.pkl'
    dictionary = defaultdict(lambda: defaultdict(list))
    start_time = time.time()
    last_log = start_time
    
    # Check for existing dictionary
    if Path(dictionary_file).exists():
        try:
            with open(dictionary_file, 'rb') as f:
                dictionary = pickle.load(f)
            logger.info(f"Loaded dictionary from {dictionary_file} with {sum(len(v) for v in dictionary.values())} chunks.")
            print(f"Loaded dictionary from {dictionary_file}")
            return dictionary
        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")
    
    # Initialize logs
    logger.info("Starting dictionary build...")
    with open('stargate_log.txt', 'w') as f:
        f.write(f"[{time.ctime()}] Starting dictionary build...\n")
    
    # Build 16-bit dictionary
    logger.info("Building 16-bit dictionary...")
    primes = [7, 11, 13, 17, 19]
    functions = [
        ('pi', get_pi),
        ('sqrt2', get_sqrt2),
        ('e', get_e),
        ('phi', get_phi),
        ('logistic_0.1', get_logistic_0_1),
        ('sin', sin_sequence),
        ('cf_sqrt2', get_cf_sqrt2),
        ('fib_7', get_fib_7)
    ]
    
    # Exhaustive 16-bit sequences
    for i in range(2**16):
        chunk = format(i, '016b')
        dictionary[16][chunk].append(f"enum.{i}.cs16")
        if i % 10000 == 0:
            elapsed = time.time() - start_time
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            logger.info(f"16-bit enum: {i} chunks, {elapsed:.2f}s, CPU: {cpu}%, Mem: {mem}%")
            print(f"Building 16-bit enum: {i} chunks, {elapsed:.2f}s")
            last_log = elapsed
        if elapsed > timeout:
            logger.warning("Timeout reached during 16-bit enum build.")
            break
    
    # Transcendental sequences
    for p in primes:
        seq = get_prime_reciprocal(p, max_digits)
        for i in range(0, min(100, len(seq) - 16 + 1), 16):  # Limit to 100
            chunk = seq[i:i+16]
            dictionary[16][chunk].append(f"1/{p}.{i}.{i+16}")
        elapsed = time.time() - start_time
        logger.info(f"Prime 1/{p}: {len(dictionary[16])} chunks, {elapsed:.2f}s")
        print(f"Prime 1/{p}: {len(dictionary[16])} chunks, {elapsed:.2f}s")
        last_log = elapsed
        if elapsed > timeout:
            logger.warning("Timeout reached during prime sequences.")
            break
    
    for func_name, func in functions:
        seq = func(max_digits)
        for i in range(0, min(100, len(seq) - 16 + 1), 16):
            chunk = seq[i:i+16]
            dictionary[16][chunk].append(f"{func_name}.{i}.{i+16}")
        elapsed = time.time() - start_time
        logger.info(f"Function {func_name}: {len(dictionary[16])} chunks, {elapsed:.2f}s")
        print(f"Function {func_name}: {len(dictionary[16])} chunks, {elapsed:.2f}s")
        last_log = elapsed
        if elapsed > timeout:
            logger.warning(f"Timeout reached during {func_name} sequences.")
            break
    
    # Save dictionary
    try:
        with open(dictionary_file, 'wb') as f:
            pickle.dump(dictionary, f)
        logger.info(f"Saved dictionary to {dictionary_file} with {len(dictionary[16])} 16-bit chunks.")
        print(f"Saved dictionary to {dictionary_file}")
        if IN_COLAB:
            files.download(dictionary_file)
    except Exception as e:
        logger.error(f"Error saving dictionary: {e}")
    
    report_dictionary(dictionary)
    
    # Generate larger sequences (on-the-fly, limited)
    for chunk_size in chunk_sizes[1:]:
        if chunk_size % 16 != 0:
            continue
        num_16bit = chunk_size // 16
        base_chunks = list(dictionary[16].keys())
        for i in range(min(50, len(base_chunks))):  # Reduced to prevent overload
            for combo in itertools.combinations_with_replacement(base_chunks[:50], num_16bit):
                chunk = ''.join(combo)
                if len(chunk) == chunk_size:
                    keys = [dictionary[16][c][0] for c in combo]
                    dictionary[chunk_size][chunk].append(f"concat.{'|'.join(keys)}")
                elapsed = time.time() - start_time
                if elapsed - last_log >= 1:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().percent
                    total_chunks = sum(len(v) for v in dictionary.values())
                    logger.info(f"Chunk Size {chunk_size}: {total_chunks} chunks, {elapsed:.2f}s, CPU: {cpu}%, Mem: {mem}%")
                    print(f"Building {chunk_size}-bit: {total_chunks} chunks, {elapsed:.2f}s")
                    last_log = elapsed
                if elapsed > timeout:
                    logger.warning(f"Timeout reached during {chunk_size}-bit sequences.")
                    break
            if elapsed > timeout:
                break
    
    elapsed = time.time() - start_time
    total_chunks = sum(len(v) for v in dictionary.values())
    logger.info(f"Dictionary setup: {elapsed:.2f}s, Chunks: {total_chunks}")
    
    # Save dictionary
    try:
        with open(dictionary_file, 'wb') as f:
            pickle.dump(dictionary, f)
        if IN_COLAB:
            files.download(dictionary_file)
    except Exception as e:
        logger.error(f"Error saving dictionary: {e}")
    
    return dictionary

def report_dictionary(dictionary):
    total_chunks = sum(len(v) for v in dictionary.values())
    logger.info(f"Dictionary Report: {total_chunks} total chunks")
    for chunk_size in dictionary:
        chunks = dictionary[chunk_size]
        functions = defaultdict(int)
        for keys in chunks.values():
            for key in keys:
                func = key.split('.')[0]
                functions[func] += 1
        logger.info(f"Chunk Size {chunk_size}: {len(chunks)} chunks")
        for func, count in functions.items():
            logger.info(f"  {func}: {count} sequences")
    print(f"Dictionary: {total_chunks} chunks (details in dictionary_build_log.txt)")

def data_to_string(data):
    return ''.join(format(b, '08b') for b in data)

def find_matches(data_str, dictionary, chunk_size=512):
    matches = []
    i = 0
    while i < len(data_str):
        chunk = data_str[i:i+chunk_size] if i+chunk_size <= len(data_str) else data_str[i:]
        if len(chunk) < chunk_size:
            break
        
        # Direct match
        if chunk in dictionary[chunk_size]:
            matches.append((dictionary[chunk_size][chunk][0], i, i+chunk_size))
            i += chunk_size
            continue
        
        # Fractal search (16-bit hierarchy)
        found = False
        if chunk_size > 16:
            num_16bit = chunk_size // 16
            base_chunks = list(dictionary[16].keys())
            for combo in itertools.combinations_with_replacement(base_chunks, num_16bit):
                candidate = ''.join(combo)
                if candidate == chunk:
                    keys = [dictionary[16][c][0] for c in combo]
                    matches.append((f"concat.{'|'.join(keys)}", i, i+chunk_size))
                    found = True
                    break
        if found:
            i += chunk_size
        else:
            # Optional 64-bit field
            matches.append((f"raw.{chunk[:64]}", i, i+64))
            i += 64
    
    return matches

def estimate_compressed_size(matches, unmatched_data, descriptor_bits=32):
    return len(matches) * descriptor_bits // 8 + len(zlib.compress(unmatched_data, level=9))

def compress_and_evaluate(data, dictionary, log_file, chunk_size=512):
    start_time = time.time()
    data_str = data_to_string(data)
    original_size = len(data)
    original_bits = len(data_str)
    
    matches = find_matches(data_str, dictionary, chunk_size=chunk_size)
    
    unmatched = bytearray()
    last_end = 0
    for _, start, end in matches:
        unmatched.extend(int(data_str[last_end:start], 2).to_bytes((start-last_end+7)//8, 'big'))
        last_end = end
    unmatched.extend(int(data_str[last_end:], 2).to_bytes((len(data_str)-last_end+7)//8, 'big'))
    
    compressed_size = estimate_compressed_size(matches, unmatched)
    ratio = compressed_size / original_size if original_size > 0 else 1.0
    
    with open(log_file, 'a') as f:
        f.write(f"\n--- Compression Report ({time.ctime()}) ---\n")
        f.write(f"Original size: {original_size} bytes\n")
        f.write(f"Compressed size: {compressed_size} bytes\n")
        f.write(f"Compression ratio: {ratio:.2f}\n")
        f.write(f"Matches found: {len(matches)}\n")
        f.write(f"Matched bits: {sum(end-start for _, start, end in matches)}/{original_bits} "
                f"({sum(end-start for _, start, end in matches)/original_bits*100:.2f}%)\n")
        f.write("Matches:\n")
        for key, start, end in matches[:10]:
            f.write(f"  {key}: {data_str[start:end][:50]}...\n")
        if len(matches) > 10:
            f.write(f"  ...and {len(matches)-10} more matches\n")
        f.write(f"Time taken: {time.time() - start_time:.2f} seconds\n")
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {ratio:.2f}")
    print(f"Matches found: {len(matches)}")
    print(f"See {log_file} for details.")
    
    return matches, unmatched, compressed_size, ratio

def select_file():
    if IN_COLAB:
        from google.colab import files
        uploaded = files.upload()
        file_path = next(iter(uploaded))
        with open(file_path, 'wb') as f:
            f.write(uploaded[file_path])
        return file_path
    else:
        try:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            root.destroy()
            return file_path
        except Exception as e:
            logger.error(f"Error in select_file: {e}")
            return None

def server_mode(dictionary, port=12345):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('localhost', port))
        server.listen(1)
        print(f"Server listening on port {port}...")
        
        conn, addr = server.accept()
        with conn:
            print(f"Connected to {addr}")
            dict_key = conn.recv(16).decode()
            print(f"Using dictionary key: {dict_key}")
            data = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            matches, unmatched = pickle.loads(data)
            
            data_str = bytearray()
            last_end = 0
            for key, start, end in matches:
                if key.startswith('concat'):
                    keys = key.split('.')[1].split('|')
                    seq = ""
                    for k in keys:
                        parts = k.split('.')
                        f_name, f_start, f_end = parts[0], int(parts[1]), int(parts[2])
                        if f_name == 'enum':
                            seq += format(int(f_start), '016b')
                        elif f_name.startswith('1/'):
                            p = int(f_name.split('/')[1])
                            seq += get_prime_reciprocal(p, 500)[f_start:f_end]
                        elif f_name in ['pi', 'sqrt2', 'e', 'phi']:
                            seq += MPMATH_CACHE.get(f_name, mpmath.mp.__dict__[f_name]().__str__().replace('.', '')[:500])[f_start:f_end]
                        elif f_name == 'sin':
                            seq += ''.join('1' if mpmath.sin(n/100) > 0 else '0' for n in range(500))[f_start:f_end]
                        elif f_name == 'logistic':
                            x0 = float(parts[1])
                            seq += logistic_map(x0, 3.9, 500)[f_start:f_end]
                        elif f_name == 'cf_sqrt2':
                            seq += continued_fraction(mpmath.mp.sqrt(2))[f_start:f_end]
                        elif f_name == 'fib_7':
                            seq += fibonacci_modulo(7, 500)[f_start:f_end]
                elif key.startswith('raw'):
                    seq = key.split('.')[1]
                else:
                    parts = key.split('.')
                    func, f_start, f_end = parts[0], int(parts[1]), int(parts[2])
                    if func == 'enum':
                        seq = format(int(f_start), '016b')
                    elif func.startswith('1/'):
                        p = int(func.split('/')[1])
                        seq = get_prime_reciprocal(p, 500)[f_start:f_end]
                    elif func in ['pi', 'sqrt2', 'e', 'phi']:
                        seq = MPMATH_CACHE.get(func, mpmath.mp.__dict__[func]().__str__().replace('.', '')[:500])[f_start:f_end]
                    elif func == 'sin':
                        seq = ''.join('1' if mpmath.sin(n/100) > 0 else '0' for n in range(500))[f_start:f_end]
                    elif func == 'logistic':
                        x0 = float(parts[1])
                        seq = logistic_map(x0, 3.9, 500)[f_start:f_end]
                    elif func == 'cf_sqrt2':
                        seq = continued_fraction(mpmath.mp.sqrt(2))[f_start:f_end]
                    elif func == 'fib_7':
                        seq = fibonacci_modulo(7, 500)[f_start:f_end]
                data_str.extend(seq.encode())
            data_str.extend(zlib.decompress(unmatched))
            
            output_file = f"received_{time.time()}.bin"
            with open(output_file, 'wb') as f:
                f.write(data_str)
            print(f"File received and saved as {output_file}")
    except Exception as e:
        logger.error(f"Error in server_mode: {e}")

def client_mode(dictionary, file_path, host='localhost', port=12345):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        matches, unmatched, _, _ = compress_and_evaluate(data, dictionary, log_file='stargate_log.txt')
        
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        
        dict_key = hashlib.md5(str(dictionary[16]).encode()).hexdigest()[:16]
        client.send(dict_key.encode())
        
        client.send(pickle.dumps((matches, unmatched)))
        client.close()
        print(f"File {file_path} sent to server.")
    except Exception as e:
        logger.error(f"Error in client_mode: {e}")

def main():
    try:
        with open('stargate_log.txt', 'w') as f:
            f.write(f"[{time.ctime()}] Starting program...\n")
        
        print("Building/loading dictionary...")
        dictionary = build_dictionary()
        
        mode = input("Enter mode (c for compression only, t for transfer): ").strip().lower()
        
        if mode == 'c':
            file_path = select_file()
            if not file_path or not Path(file_path).exists():
                print("No valid file selected, exiting.")
                logger.error("No valid file selected.")
                return
            with open(file_path, 'rb') as f:
                data = f.read()
            compress_and_evaluate(data, dictionary, log_file='stargate_log.txt')
        
        elif mode == 't':
            transfer_mode = input("Run as server (s) or client (c)? ").strip().lower()
            if transfer_mode == 's':
                server_mode(dictionary)
            elif transfer_mode == 'c':
                file_path = select_file()
                if not file_path or not Path(file_path).exists():
                    print("No valid file selected, exiting.")
                    logger.error("No valid file selected.")
                    return
                client_mode(dictionary, file_path)
            else:
                print("Invalid transfer mode, exiting.")
                logger.error("Invalid transfer mode.")
        
        else:
            print("Invalid mode, exiting.")
            logger.error("Invalid mode.")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
