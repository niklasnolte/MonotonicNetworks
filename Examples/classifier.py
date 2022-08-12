import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import matplotlib.pyplot as plt

from monotonenorm import direct_norm, SigmaNet, GroupSort

torch.manual_seed(42)

monotonic = True  # want to be monotonic?
LIP = 3  # lipschitz constant of the model
EPOCHS = 500


def get_data(nsamples = 6000):
    x0_0 = torch.normal(-1.0, 1.0, (nsamples * 3 // 4, 1))
    x0_1 = torch.normal(2.0, 0.5, (nsamples // 4, 1))  # malicious data
    x0 = torch.vstack((x0_0, x0_1))
    y0 = torch.normal(-0.5, 0.5, (nsamples, 1))
    class0 = torch.hstack((x0, y0))
    class1 = torch.normal(1, 0.5, (nsamples, 2))

    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([torch.zeros(class0.shape[0]), torch.ones(class1.shape[0])])
    return X, y


def get_model(monotonic):
  def lipschitz_norm(module):
      return direct_norm(
          module,  # the layer to constrain
          "one",  # |W|_1 constraint type
          max_norm=LIP ** (1 / 3),  # norm of the layer (LIP ** (1/nlayers))
      )

  model = torch.nn.Sequential(
      lipschitz_norm(torch.nn.Linear(2, 8)),
      GroupSort(2),
      lipschitz_norm(torch.nn.Linear(8, 8)),
      GroupSort(2),
      lipschitz_norm(torch.nn.Linear(8, 1)),
  )

  if monotonic:
      model = SigmaNet(
          model,
          sigma=LIP,
          monotone_constraints=[1, 1,],
          # 0: don't constrain feature monotonicity,
          # 1: monotonically increasing,
          # -1: monotonically decreasing
          # for each feature individually
      )
  return model


#training
X, y = get_data()
model = get_model(monotonic)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


for i in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = binary_cross_entropy_with_logits(y_pred, y.view(-1, 1))
    loss.backward()
    optimizer.step()

# evaluation (on grid for decision frontier visualization)
range_x0 = (X[:, 0].max(), X[:, 0].min())
range_x1 = (X[:, 1].max(), X[:, 1].min())

x0mesh, x1mesh = torch.meshgrid(
    torch.linspace(*range_x0, 20), torch.linspace(*range_x1, 20), indexing="ij"
)
gridinput = torch.stack([x0mesh, x1mesh], axis=-1)
with torch.no_grad():
    gridoutput = torch.sigmoid(model(gridinput))
    gridoutput = gridoutput.view(x0mesh.shape).numpy()

c = plt.contour(x0mesh, x1mesh, gridoutput, levels=10)
plt.clabel(c, c.levels, inline=True, fontsize=10)

plt.scatter(*X[y == 0].T.numpy())
plt.scatter(*X[y == 1].T.numpy())
plt.show()
