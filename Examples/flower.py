import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from monotonenorm import GroupSort, direct_norm
torch.manual_seed(0)
plt.style.use('mpl-config')

# The following is a simple example to show that Lipschitz-constrained networks can
# describe arbitrarily complex decision boundaries.
# We will train a Lipschitz network on a toy 2D dataset.

# --------------- Configure the training/dataset ---------------
n = 150   # Number of points per radius
nrads = 3  # number of radii to use in training
freq = 11  # Frequency of the sine wave used to discribe the boundary

EPOCHS = 300
SHUFFLE = True
BATCH_SIZE = 16
loss_func = torch.nn.L1Loss()
PREFIX = "L1"
LR_INIT = 1e-2
LR_FINAL = 1e-4
STEPS = EPOCHS * n * nrads // BATCH_SIZE
GAMMA = (LR_FINAL / LR_INIT)**(1 / STEPS)

theta = np.linspace(0, np.pi, n)
points = []  # Coordinates of points to train on
rs = []      # Regression targets of the points
for r in np.linspace(1, 10, nrads):
    r = r * np.ones_like(theta)  # regression target
    rs.extend([r, r])
    r = r * (1 - 0.18 * np.cos(freq * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points.append(
        np.hstack(
            [np.concatenate([y, -y])[:, None],
             np.concatenate([x, x])[:, None]]))
train_data = np.vstack(points)
train_labels = np.concatenate(rs)
train_labels /= 2  # To ensure that the regression function is Lipschitz-1.
x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    shuffle=SHUFFLE, batch_size=BATCH_SIZE)

# Plot training data
# sc = plt.scatter(*x_train.T, c=y_train, s=1)
# plt.colorbar(sc)
# plt.show()
# exit()


# --------------- Build Lipschitz-1 Network ---------------
model = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(2, 1024), kind="two-inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(1024, 1024), kind="inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(1024, 256), kind="inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(256, 256), kind="inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(256, 32), kind="inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(32, 32), kind="inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(32, 32), kind="inf"),
    GroupSort(16),
    direct_norm(torch.nn.Linear(32, 1), kind="inf"),
)
optim = torch.optim.Adam(model.parameters(), lr=LR_INIT)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=GAMMA)

# --------------- Train and save the Network ---------------
pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    for x, y in train_loader:
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step()
        msg = f"epoch: {epoch} loss: {loss.item():.4f}"
        pbar.set_description(msg)

model_name = f"model_{loss.item():.2e}.pt"
if BATCH_SIZE != len(x_train):
    model_name = model_name.replace(".pt", f"_bs{BATCH_SIZE}.pt")
torch.save(model.state_dict(), model_name)

# --------------- Plot the results ---------------
n_grid = 200
# make a grid of points to evaluate the network
X, Y = np.meshgrid(*[np.linspace(1.05 * x_train.min(), 1.05 * x_train.max(), n_grid)] * 2)
grid = torch.tensor(np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))).float()
with torch.no_grad():
    model.eval()
    y_pred = model(grid)
    y_pred = y_pred.numpy().reshape(n_grid, n_grid)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
plt.pcolormesh(X, Y, y_pred, alpha=1)  # heatmap of the grid
plt.colorbar()
# contour lines of the network
cs = plt.contour(X, Y, y_pred, colors="k", linestyles="solid")
plt.clabel(cs, inline=True, fontsize=10)

plt.scatter(*x_train.T, c="w", s=1, alpha=0.5)  # overlay the training data
fig_name = f"flower.png"
if BATCH_SIZE != len(x_train):
    fig_name = fig_name.replace(".png", f"_bs{BATCH_SIZE}.png")
plt.savefig(fig_name)
print(fig_name, "saved")
