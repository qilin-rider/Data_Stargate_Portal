# Section 0: Imports and Setup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

# Constants
INPUT_SIZE = 256  # Input sequence length in bits
MAX_PARAM_SIZE = 80  # Max parameters; variable length below this
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SPARSITY_WEIGHT = 0.01

# Section 1: Neural Network Definitions
class EncoderNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, max_param_size=MAX_PARAM_SIZE):
        super(EncoderNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=10, padding=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=10, padding=5)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(32, 64, batch_first=True, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(64 * 2, 256)
        self.fc2 = nn.Linear(256, max_param_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_size]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))  # [batch, max_param_size]
        return x

class DecoderNet(nn.Module):
    def __init__(self, max_param_size=MAX_PARAM_SIZE, output_size=INPUT_SIZE):
        super(DecoderNet, self).__init__()
        self.fc1 = nn.Linear(max_param_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Section 2: Autoencoder Class
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Section 3: Data Generation
def generate_patterned_sequence(length=INPUT_SIZE):
    patterns = [
        lambda: [0] * length,                            # All zeros
        lambda: [1] * length,                            # All ones
        lambda: [i % 2 for i in range(length)],         # Alternating
        lambda: [random.choice([0, 1]) for _ in range(length)],  # Random
        lambda: [0, 0, 1, 1] * (length // 4),           # Repeated block
        lambda: [(i % 8) < 4 for i in range(length)],   # Arithmetic
        lambda: [int(i % 5 == 0) for i in range(length)],  # Sparse
        lambda: [0] * (length // 2) + [1] * (length // 2),  # Half split
    ]
    seq = random.choice(patterns)()
    return seq[:length]

train_data = [generate_patterned_sequence() for _ in range(10000)]
train_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in train_data]

# Section 4: Variable-Length Rule-Based Compression
def compress_params(encoded):
    """Convert fixed-size encoded params to variable-length based on rules."""
    encoded = np.round(encoded).astype(int)
    # Rule 1: All same value
    if np.all(encoded == encoded[0]):
        return (encoded[0],)  # e.g., {1} for all ones
    # Rule 2: Alternating pair
    test_seq = np.tile(encoded[:2], MAX_PARAM_SIZE // 2)[:MAX_PARAM_SIZE]
    if np.array_equal(encoded, test_seq):
        return tuple(encoded[:2])  # e.g., {0, 1} for alternating
    # Default: Full params
    return tuple(encoded)

def decompress_params(params, target_length=INPUT_SIZE):
    """Expand variable-length params to full sequence."""
    if len(params) == 1:
        return [params[0]] * target_length  # Repeat single value
    elif len(params) == 2:
        return list(np.tile(params, target_length // 2))[:target_length]  # Repeat pair
    else:
        # For now, assume full params; later, add more rules
        decoded = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
        return decoder(decoded).detach().numpy()[0]

# Section 5: Training Loop
def train_autoencoder(autoencoder, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            batch_tensor = torch.stack(batch)
            optimizer.zero_grad()
            encoded, decoded = autoencoder(batch_tensor)
            recon_loss = criterion(decoded, batch_tensor)
            sparsity_loss = SPARSITY_WEIGHT * torch.mean(torch.abs(encoded))
            loss = recon_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i // batch_size + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {(i//batch_size)+1}, "
                      f"Loss: {loss.item():.6f}, Compression Ratio: {INPUT_SIZE/MAX_PARAM_SIZE:.2f}x")
        avg_loss = total_loss / (len(train_data) / batch_size)
        print(f"Epoch {epoch+1}/{epochs} Summary - Avg Loss: {avg_loss:.6f}")

# Section 6: Library Construction
def build_library(autoencoder, train_data):
    library = {}
    tricky_list = {}
    for seq in train_data:
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        encoded = autoencoder.encoder(seq_tensor).detach().numpy()[0]
        param_key = compress_params(encoded)
        if param_key in library and not np.array_equal(library[param_key], seq):
            tricky_id = f"tricky{len(tricky_list) + 1}"
            tricky_list[tricky_id] = seq
        else:
            library[param_key] = seq
    return library, tricky_list

# Section 7: Encode and Decode Functions
def encode(seq, autoencoder, library, tricky_list):
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    encoded = autoencoder.encoder(seq_tensor).detach().numpy()[0]
    param_key = compress_params(encoded)
    if param_key in library and np.array_equal(library[param_key], seq):
        return param_key
    else:
        for tid, tseq in tricky_list.items():
            if np.array_equal(tseq, seq):
                return tid
        tricky_id = f"tricky{len(tricky_list) + 1}"
        tricky_list[tricky_id] = seq
        return tricky_id

def decode(code, library, tricky_list):
    if isinstance(code, str) and code.startswith("tricky"):
        return tricky_list.get(code, None)
    seq = library.get(code, None)
    if seq is None and isinstance(code, tuple):
        return decompress_params(code)
    return seq

# Section 8: Testing
def test_compression(autoencoder, library, tricky_list):
    test_seq = [1] * INPUT_SIZE  # All ones
    code = encode(test_seq, autoencoder, library, tricky_list)
    decoded = decode(code, library, tricky_list)
    print(f"All Ones: Original={test_seq[:10]}..., Code={code}, "
          f"Decoded={decoded[:10] if decoded is not None else 'None'}...")

    test_seq = [0, 1] * (INPUT_SIZE // 2)  # Alternating
    code = encode(test_seq, autoencoder, library, tricky_list)
    decoded = decode(code, library, tricky_list)
    print(f"Alternating: Original={test_seq[:10]}..., Code={code}, "
          f"Decoded={decoded[:10] if decoded is not None else 'None'}...")

# Main Execution
if __name__ == "__main__":
    encoder = EncoderNet()
    decoder = DecoderNet()
    autoencoder = AutoEncoder(encoder, decoder)

    train_autoencoder(autoencoder, train_tensors)
    library, tricky_list = build_library(autoencoder, train_data)
    test_compression(autoencoder, library, tricky_list)
