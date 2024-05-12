import torch
import torch.nn as nn
import numpy as np

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

# Prepare the input data for GRU
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
    padded_events = np.pad(seq[0], (0, max_length - len(seq[0]))).tolist()
    padded_times = np.pad(seq[1], (0, max_length - len(seq[1]))).tolist()
    padded_sequences.append([padded_events, padded_times])

# Convert sequences to PyTorch tensors
padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)


# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        output = output[:, -1, :]  # Take the last GRU output
        output = self.fc(output)
        return output


input_size = 2  # Two input features: violation type and time
hidden_size = 32
output_size = 1  # Output layer with sigmoid activation for binary classification

model = GRUModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 50
targets = torch.ones(padded_sequences.size(0), dtype=torch.float32)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(padded_sequences[:, :, :2])  # Only pass the first two features to the model
    loss = criterion(outputs.squeeze(), targets)
    loss.backward()
    optimizer.step()

# Get the GRU representations for each company's violation events
gru_representations = model(padded_sequences[:, :, :2])  # Only pass the first two features

# Print the GRU representations
for i, gru_representation in enumerate(gru_representations):
    company_code = data[i][0][0]  # Get the company code
    print(f"Company {company_code} GRU representation: {gru_representation.tolist()}")