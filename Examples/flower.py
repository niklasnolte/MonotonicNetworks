import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from monotonenorm import GroupSort, direct_norm
from argparse import ArgumentParser
from pathlib import Path

torch.manual_seed(0)
# plt.style.use("mpl-config")

parser = ArgumentParser()
parser.add_argument(
    "--train",
    action="store_true",
    help="Force the model to re-train even if a saved model exists",
)
args = parser.parse_args()
# The following is a simple example to show that Lipschitz-constrained networks can
# describe arbitrarily complex decision boundaries.
# We will train a Lipschitz network on a toy 2D dataset.

# --------------- Configure the training/dataset ---------------
n = 200  # Number of points per radius
nrads = 3  # number of radii to use in training
freq = 20  # Frequency of the sine wave used to discribe the boundary

EPOCHS = 2000  # Train for longer for a better result (duh)
SHUFFLE = True
BATCH_SIZE = 2**15
loss_func = torch.nn.L1Loss()
# LR_INIT = 1e-2
# LR_FINAL = 5e-3
LR_INIT = 3e-2
LR_FINAL = 1e-2

theta = np.linspace(0, np.pi, n)
points = []  # Coordinates of points to train on
rs = []  # Regression targets of the points
for r in np.linspace(1, 2, nrads):
    r = r * np.ones_like(theta)  # regression target
    rs.extend([r, r])
    r = r * (1 - 0.1 * np.cos(freq * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points.append(
        np.hstack(
            [np.concatenate([y, -y])[:, None], np.concatenate([x, x])[:, None]]
        )
    )
train_data = np.vstack(points)
train_labels = np.concatenate(rs)
train_labels -= 1
x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(
    TensorDataset(x_train, y_train), shuffle=SHUFFLE, batch_size=BATCH_SIZE
)

STEPS = EPOCHS * len(train_loader)

max_norm = 3 ** (1 / 8)
# --------------- Build Lipschitz Network ---------------
model = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(2, 1024), kind="two-inf", max_norm=max_norm),
    torch.nn.BatchNorm1d(1024, affine=False),
    GroupSort(16),
    direct_norm(torch.nn.Linear(1024, 1024), kind="inf", max_norm=max_norm),
    GroupSort(16),
    direct_norm(torch.nn.Linear(1024, 256), kind="inf", max_norm=max_norm),
    GroupSort(16),
    direct_norm(torch.nn.Linear(256, 256), kind="inf", max_norm=max_norm),
    GroupSort(16),
    direct_norm(torch.nn.Linear(256, 32), kind="inf", max_norm=max_norm),
    GroupSort(16),
    direct_norm(torch.nn.Linear(32, 32), kind="inf", max_norm=max_norm),
    GroupSort(16),
    direct_norm(torch.nn.Linear(32, 32), kind="inf", max_norm=max_norm),
    GroupSort(16),
    direct_norm(torch.nn.Linear(32, 1), kind="inf", max_norm=max_norm),
)

optim = torch.optim.Adam(model.parameters(), lr=LR_FINAL)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=LR_INIT, steps_per_epoch=len(train_loader), epochs=EPOCHS
)
# --------------- Train and save the Network ---------------
model_name = "model.pt"
path_root = Path(__file__).parent
model_path = path_root / model_name
if args.train or not Path(model_path).exists():
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
    torch.save(model.state_dict(), model_path)
    print("Saved to:", model_path)

# --------------- Load the Network ---------------
model.load_state_dict(torch.load(model_path))
model.eval()

# --------------- Plot the results ---------------
n_grid = 1000
# make a grid of points to evaluate the network
X, Y = np.meshgrid(
    *[np.linspace(1.0 * x_train.min(), 1.0 * x_train.max(), n_grid)] * 2
)
grid = torch.tensor(np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))).float()
with torch.no_grad():
    model.eval()
    y_pred = model(grid)
    y_pred = y_pred.numpy().reshape(n_grid, n_grid)
    y_train_pred = model(x_train).numpy().flatten()

fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
c = y_pred  # (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
# plt.pcolormesh(X, Y, c, alpha=1)  # heatmap of the grid

plt.imshow(
    c,
    extent=[X.min(), X.max(), Y.min(), Y.max()],
    origin="lower",
    alpha=1,
    cmap="Blues_r",
    vmax=1,
    vmin=0.03,
)
plt.colorbar()

levels = [0.5, 1]
colors = ["royalblue"] * len(levels)
colors[levels.index(0.5)] = "red"
cs = plt.contour(
    X,
    Y,
    y_pred,
    colors=colors,
    linestyles="solid",
    levels=levels,
    alpha=0.9,
    zorder=10,
    linewidths=2,
)
plt.clabel(cs, inline=True, fontsize=10, fmt="%.2f")


plt.scatter(
    *x_train[::2].T, c="black", s=20, alpha=1, marker="x", zorder=11  # type: ignore
)  # overlay the training data
fig_name = "flower.jpg"
plt.tight_layout(pad=0)

file_path = Path.joinpath(path_root, "figures/")
file_path.mkdir(exist_ok=True)
file_path = Path.joinpath(file_path, fig_name)
plt.savefig(file_path, dpi=300)
print("Saved to:", file_path)
