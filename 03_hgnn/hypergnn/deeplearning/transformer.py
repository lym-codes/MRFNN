import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
# Define the input data
data = [
    # Company 1's violation event sequence
    [
        ["001", 9901, 0, 999],
        ["001", 9902, 1, 888],
        ["001", 9903, 0, 777]
    ],
    # Company 2's violation event sequence
    [
        ["002", 9904, 2, 666],
        ["002", 9905, 0, 555],
        ["002", 9905, 1, 444]
    ]
]

# Prepare the input data for Transformer
sequences = []
for company_events in data:
    events = []
    times = []
    for event in company_events:
        events.append(event[2])  # Use violation type as input feature
        times.append(event[3])  # Use violation time as input feature
    sequences.append([events, times])  # Combine violation types and times

# Find the maximum sequence length
max_length = max([len(seq[0]) for seq in sequences])

# Pad sequences to the maximum length
padded_sequences = []
for seq in sequences:
    padded_events = torch.tensor(seq[0]).unsqueeze(0)
    padded_times = torch.tensor(seq[1]).unsqueeze(0)
    padded_events = F.pad(padded_events, (0, max_length - len(seq[0])), value=0)
    padded_times = F.pad(padded_times, (0, max_length - len(seq[1])), value=0)
    padded_sequences.append(torch.cat([padded_events, padded_times], dim=0))

# Pad the sequences to form a batch
padded_sequences = pad_sequence(padded_sequences, batch_first=True)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.transformer = nn.Transformer(d_model=input_size, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, src, tgt):
        src = src.transpose(0, 1)  # Transpose the dimensions of the source input tensor
        tgt = tgt.transpose(0, 1)  # Transpose the dimensions of the target input tensor
        output = self.transformer(src, tgt)
        output = output[-1, :, :]  # Take the output of the last time step
        output = self.fc(output)
        return output

# Create the Transformer model instance
input_size = 2  # Two input features: violation type and time
hidden_size = 32
output_size = 1  # Output layer with sigmoid activation for binary classification
num_layers = 2  # Number of layers in the Transformer encoder
num_heads = 1  # Number of attention heads

model = TransformerModel(input_size, hidden_size, output_size, num_layers, num_heads)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 100
targets = torch.ones(padded_sequences.size(0), dtype=torch.float32)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(padded_sequences[:, :, :2], padded_sequences[:, :, :2])  # Pass source and target inputs to the model
    loss = criterion(outputs.squeeze(), targets)
    loss.backward()
    optimizer.step()

# Get the Transformer representations for each company's violation events
transformer_representations = model(padded_sequences[:, :, :2], padded_sequences[:, :, :2])  # Pass source and target inputs to the model

# Print the Transformer representations
for i, transformer_representation in enumerate(transformer_representations):
    company_code = data[i][0][0]  # Get the company code
    print(f"Company {company_code} Transformer representation: {transformer_representation.tolist()}")