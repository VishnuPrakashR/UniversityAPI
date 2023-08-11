import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import os

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
        )
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        residual = x
        x = self.attention(x, x, x)[0]
        x = self.layer_norm1(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm2(x + residual)

        return x

class TransformerSentenceBuilder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerSentenceBuilder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_size, hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = self.positional_encoding(embedded)

        for transformer_block in self.transformer_blocks:
            embedded = transformer_block(embedded)

        logits = self.linear(embedded)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_length=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def preprocess_input(prompt, tokenizer, max_length):
    # Tokenize the prompt
    tokens = tokenizer(prompt.lower())

    # Build vocabulary mapping from tokens
    vocab = build_vocab_from_iterator([tokens])

    # Convert tokens to integers using the vocabulary mapping
    input_ids = [vocab[token] for token in tokens if token in vocab]

    # Pad or truncate the input_ids to the specified max_length
    if len(input_ids) < max_length:
        padded_input_ids = input_ids + [0] * (max_length - len(input_ids))
    else:
        padded_input_ids = input_ids[:max_length]

    # Convert the padded_input_ids list to a tensor
    input_tensor = torch.tensor(padded_input_ids).unsqueeze(0)  # Add batch dimension

    return input_tensor

def preprocess_target(response, tokenizer, max_length):
    # Tokenize the response
    tokens = tokenizer(response.lower())

    # Build vocabulary mapping from tokens
    vocab = build_vocab_from_iterator([tokens])

    # Convert tokens to integers using the vocabulary mapping
    target_ids = [vocab[token] for token in tokens if token in vocab]

    # Pad or truncate the target_ids to the specified max_length
    if len(target_ids) < max_length:
        padded_target_ids = target_ids + [0] * (max_length - len(target_ids))
    else:
        padded_target_ids = target_ids[:max_length]

    # Convert the padded_target_ids list to a tensor
    target_tensor = torch.tensor(padded_target_ids).unsqueeze(0)  # Add batch dimension

    return target_tensor

def post_process_output(output, tokenizer, user_prompt):
    predicted_indices = torch.argmax(output, dim=2).squeeze(0).tolist()
    vocab = build_vocab_from_iterator([tokenizer(user_prompt.lower())], specials=["<unk>"])
    print(vocab, predicted_indices)
    predicted_words = [vocab.itos[idx] for idx in predicted_indices if idx > 0]
    generated_text = " ".join(predicted_words)
    return generated_text

def load_weights(model, weights_path):
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print("Model weights loaded successfully.")
    else:
        print("Weights file does not exist. Initializing model from scratch.")

def save_weights(model, weights_path):
    torch.save(model.state_dict(), weights_path)
    print("Model weights saved successfully.")

def main():
    # Define hyperparameters
    vocab_size = 10000
    embedding_size = 256
    hidden_size = 512
    num_layers = 4
    num_heads = 8
    dropout = 0.1
    learning_rate = 0.001
    max_length = 10

    # Instantiate the model
    model = TransformerSentenceBuilder(vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Load or train the model weights
    load_weights(model, 'model_weights.pth')  # Custom function to load weights from file or train from scratch

    # Training loop
    while True:
        user_prompt = input("Enter a jumbled sentence prompt (or 'quit' to exit): ")
        if user_prompt.lower() == 'quit':
            break

        # Convert user prompt to input tensor
        input_data = preprocess_input(user_prompt, tokenizer, max_length)

        # Forward pass
        output = model(input_data)

        # Post-process the output
        generated_text = post_process_output(output, tokenizer, user_prompt)
        print("Generated Text:", generated_text)

        # Prompt user for correct response
        correct_response = input("Enter the correct sentence: ")

        # Convert correct response to target tensor
        target = preprocess_target(correct_response, tokenizer, max_length)

        # Calculate loss
        loss = criterion(output.squeeze(), target.squeeze())

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

        # Save the updated weights to a file
    save_weights(model, 'model_weights.pth')

    print("Training completed.")

if __name__ == "__main__":
    main()
