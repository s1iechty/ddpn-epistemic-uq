import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

# --- IMPORT YOUR CLASS ---
# This assumes your file is named 'vi_nn.py' inside 'models/bayesian_uq'
from deep_uncertainty.models.bayesian_uq.vi_nn import DoublePoissonVINN

# ==========================================
# 1. Generate Fake Data
# ==========================================
torch.manual_seed(42)
n_points = 400
x_all = torch.rand(n_points) * 10 - 3  
# Create a gap in the data to test uncertainty
x_train = torch.cat([
    x_all[(x_all < 0) | (x_all > 2)],       # Dense regions
    x_all[(x_all >= 0) & (x_all <= 2)][::5] # Sparse region (gap)
]).unsqueeze(1)

# Function: exp(0.7x - 0.05x^2 + 1.0)
lambda_x = torch.exp(0.7 * x_train - 0.05 * x_train**2 + 1.0)
y_train = torch.poisson(lambda_x)

# Create DataLoader
dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

# ==========================================
# 2. Initialize Your Model
# ==========================================
model = DoublePoissonVINN(
    num_data=len(x_train),
    input_size=1,
   # output_dim=2, 
    layers=[64, 64], 
    lr=1e-3,
    build_params={"init_log_sds": -5, "temperature": 0.01}
)

# ==========================================
# 3. Train
# ==========================================
trainer = pl.Trainer(max_epochs=800, accelerator='cpu', enable_progress_bar=True)
print("Starting Training...")
trainer.fit(model, dataloader)

# ==========================================
# 4. Visualize
# ==========================================
print("Plotting results...")
x_test = torch.linspace(-3, 7, 200).unsqueeze(1)

# Use the predict method (inherited from the Base Class)
# This handles the Monte Carlo sampling automatically
preds = model.predict(x_test) 

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, alpha=0.5, color='black', label='Data')
plt.plot(x_test, preds, color='blue', label='Mean Prediction')
plt.title("Double Poisson VI Results")
plt.legend()
plt.show()