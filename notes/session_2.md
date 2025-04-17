# Day 1 – Session 2 Instructor Notes

**Duration**: 3 hours

---

## Overview

This session focuses on:

1. **Recap of Session 1**
2. **Intro to PyTorch Tensors and Modules**
3. **LightningModule and LightningDataModule**
4. **Minimal Working Classifier**
5. **Intro to Multi-task Learning**
6. **Training with PyTorch Lightning**
7. **Logging & Visualization**
8. **Prep for Bounding Box Integration**

### Learning Outcomes

By the end of this session, participants should be able to:
- Understand PyTorch's core tensor and module concepts.
- Build a LightningModule and LightningDataModule.
- Train and evaluate a simple classification model using Lightning.
- Understand multi-task learning and model head separation.
- Use built-in Lightning logging for monitoring training.

---

## 1. Recap of Session 1 (10–15 min)

### Key Talking Points
- Briefly revisit what we covered: project structure, git, HPC setup.
- Confirm environment is ready and data is downloaded.
- Clarify what we’ll achieve in this session: a working classifier and first look at multi-task setup.

### Instructor Tips
- Use this time to answer setup questions or debug git/HPC issues.
- Live-demo navigating the project structure if needed.

---

## 2. Introduction to PyTorch Tensors and Modules (35–40 min)

### Key Concepts
- **`torch.Tensor`** – primary data structure
- **Tensor operations**: indexing, shapes, broadcasting
- **`nn.Module`** – base class for building models
- **PyTorch workflow**: data handling, model definition, and training loop

### The Three Essential Components of a PyTorch Workflow

A typical PyTorch project involves three key components:

1. **Data Class**: Organizes and prepares the dataset.
2. **Model Class**: Defines the neural network architecture.
3. **Training Code**: Specifies how the model learns from data.

### Example Implementation

Let's create a complete but minimal example:

#### 1. Data Class

```python
import torch
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

#### 2. Model Class

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)
```

#### 3. Training Code

```python
import torch.optim as optim

# Initialize dataset and dataloader
dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):  # simple 5-epoch example
    total_loss = 0
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/5], Loss: {avg_loss:.4f}")
```

### Live Coding Guidance
- Walk through each component in a Python shell or a Jupyter notebook.
- Emphasize how tensors flow through the data loader, model, and loss function.

### Learning Outcome
Participants should be comfortable creating tensors, writing basic dataset and model classes, and implementing a simple training loop.


---

## 3. Enhancing PyTorch Workflow with PyTorch Lightning (25–30 min)

### Key Concepts
- **PyTorch Lightning** – simplifies and structures PyTorch code
- Lightning modules: structured training loops, validation, and logging
- Lightning trainers: manage training loops, GPUs, and distributed training

### Converting to PyTorch Lightning

Let's enhance our previous example by introducing PyTorch Lightning:

#### 1. Lightning Data Module

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class RandomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_samples=100, input_dim=10):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_dim = input_dim

    def setup(self, stage=None):
        self.train_dataset = RandomDataset(self.num_samples, self.input_dim)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
```

#### 2. Lightning Module (Model + Training Logic)

```python
import torch.nn.functional as F

class SimpleLightningNet(pl.LightningModule):
    def __init__(self, input_dim=10, output_dim=2, lr=0.01):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.lr = lr

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)
```

#### 3. Lightning Trainer

```python
# Initialize data module and model
data_module = RandomDataModule()
model = SimpleLightningNet()

# Initialize trainer and run training
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, datamodule=data_module)
```

### Live Coding Guidance
- Highlight the simplicity Lightning brings to the training process.
- Emphasize reduced boilerplate and structured code management.

### Learning Outcome
Participants should understand the benefits of using PyTorch Lightning and feel comfortable converting basic PyTorch workflows to Lightning.

## 4. Leveraging PyTorch Lightning Callbacks (15–20 min)

### Key Concepts
- **Callbacks** – automated hooks to extend trainer functionality
- Common callbacks: checkpointing, early stopping, logging

### Demonstrating Callbacks

Enhance training with callbacks using PyTorch Lightning:

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Initialize data module and model
data_module = RandomDataModule()
model = SimpleLightningNet()

# Define callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min'
)

early_stopping_callback = EarlyStopping(
    monitor='train_loss',
    patience=3,
    mode='min'
)

# Initialize trainer with callbacks
trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Train the model
trainer.fit(model, datamodule=data_module)
```
