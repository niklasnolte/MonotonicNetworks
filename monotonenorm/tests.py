# Bug fix for project_norm
import unittest
from monotonenorm import GroupSort, project_norm, direct_norm
import torch


class TestNorms(unittest.TestCase):
    def test_no_norm(self):
        loss = train(lambda x, **kwargs: x)
        self.assertAlmostEqual(loss, 0.0, places=4)

    def test_project_norm(self):
        loss = train(project_norm)
        self.assertAlmostEqual(loss, 0.0, places=4)

    def test_direct_norm(self):
        loss = train(direct_norm)
        self.assertAlmostEqual(loss, 0.0, places=4)


def train(norm):
    model = torch.nn.Sequential(
        norm(torch.nn.Linear(1, 32), kind="two-inf", always_norm=False),
        GroupSort(2),
        norm(torch.nn.Linear(32, 32), kind="inf", always_norm=False),
        GroupSort(2),
        norm(torch.nn.Linear(32, 1), kind="inf", always_norm=False),
    )
    # define training target
    # fit f(x) = x^2 / 2
    x = torch.linspace(-1, 1, 100).view(-1, 1)
    y = x**2 / 2

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(2000):
        loss = torch.mean((model(x) - y)**2)
        loss.backward()
        optim.step()
        optim.zero_grad()
    return loss.item()


if __name__ == '__main__':
    torch.manual_seed(0)
    # run the tests
    unittest.main()
