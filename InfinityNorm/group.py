import torch
from torch import nn


class GroupSort(nn.Module):
    def __init__(self, num_units, axis=-1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        return group_sort(x, self.num_units, self.axis)

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def process_group_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis + 1, num_channels // num_units)
    return size


def group_sort(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))
    return sorted_x
