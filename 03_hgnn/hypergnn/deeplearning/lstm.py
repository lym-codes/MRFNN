import torch
import torch.nn as nn
import numpy as np

# Define the input data
data = [
    # Company 1's violation event sequence
    [# 第一列是企业代码； 第二列是违规类别；第三列：违规次数/月；第四列，时间（月份数字）
        ["001", 101, 1, 5],
        ["001", 102, 1, 7],
        ["001", 101, 2, 6],
        ["001", 102, 2, 9],
        ["001", 101, 3, 8],
        ["001", 102, 3, 12]
    ],
    # Company 2's violation event sequence
    [
        ["002", 101, 1, 2],
        ["002", 103, 1, 12],
        ["002", 101, 2, 11],
        ["002", 102, 2, 8],
        ["002", 101, 3, 6],
        ["002", 102, 3, 5]
    ]
]

# Prepare the input data for LSTM
sequences = []
for company_events in data:
    events = []
    times = []
    counts = []
    for event in company_events:
        events.append(event[1])  # Use violation type as input feature
        times.append(event[2])  # Use violation time as input feature
        counts.append(event[3])
    sequences.append([events, times,counts])  # Combine violation types and times

# Find the maximum sequence length
max_length = max([len(seq[0]) for seq in sequences])

# Pad sequences to the maximum length
padded_sequences = []
for seq in sequences:
    padded_events = np.pad(seq[0], (0, max_length - len(seq[0]))).tolist()
    padded_times = np.pad(seq[1], (0, max_length - len(seq[1]))).tolist()
    padded_counts = np.pad(seq[2], (0, max_length - len(seq[2]))).tolist()
    padded_sequences.append([padded_events, padded_times,padded_counts])

# Convert sequences to PyTorch tensors
padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Take the last LSTM output
        output = self.fc(output)
        return output


input_size = 3  # Two input features: violation type and time
hidden_size = 32
output_size = 1  # Output layer with sigmoid activation for binary classification

model = LSTMModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 50
targets = torch.ones(padded_sequences.size(0), dtype=torch.float32)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # 第一个企业；第二个是字段，第三个是排行
    outputs = model(padded_sequences[:, :, :3])  # Only pass the first two features to the model
    loss = criterion(outputs.squeeze(), targets)
    loss.backward()
    optimizer.step()
# 第一个方括号是企业，第2个
# Get the LSTM representations for each company's violation events
lstm_representations = model(padded_sequences[:, :, :3])  # Only pass the first two features

# Print the LSTM representations
for i, lstm_representation in enumerate(lstm_representations):
    company_code = data[i][0][0]  # Get the company code
    print(f"Company {company_code} LSTM representation: {lstm_representation.tolist()}")
