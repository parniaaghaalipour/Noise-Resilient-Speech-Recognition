import torch
import numpy as np
from torch.optim import SGD, lr_scheduler
from torch.nn import CTCLoss
from data_loader import DataLoader  # Custom data loader
from decoder import GreedyDecoder
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

# Parsing command-line options
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# Setting seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
data_loader = DataLoader(batch_size=args.batch_size)
data_loader.load_data('train_path', 'test_path')  # Setting actual data paths

# Creating model 
model = TransformerModel(input_dim, num_heads, num_layers, dropout, output_dim)
model = model.to(device)

# For multiple gpus
if torch.cuda.device_count() > 1:
    model = DistributedDataParallel(model)

# Setting optimizer
optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, 
                weight_decay=args.weight_decay)

# Scheduling learning rate
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Loss function
criterion = CTCLoss()

tb_writer = SummaryWriter()

# Training loop
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(data_loader.train_data):

        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        output = output.transpose(0, 1)  # For CTC Loss, expected shape is (T, N, C)
        
        # Decoding output - The example uses Greedy decoder as an example
        decoded_output = GreedyDecoder(output)
        
        loss = criterion(decoded_output, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # print statistics
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # Write the training loss to tensorboard
    tb_writer.add_scalar('Loss/train', running_loss, epoch)
    # Step the scheduler
    scheduler.step()

        
# Saving the trained model
torch.save(model.state_dict(), 'model.pth')

tb_writer.close()
