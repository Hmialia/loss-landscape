"""
    Serialization and deserialization of directions in the direction file.
"""

import torch
import numpy as np # 确保导入

def write_list(f, name, direction):
    """ Save the direction to the hdf5 file with name as the key

        Args:
            f: h5py file object
            name: key name_surface_file
            direction: a list of tensors
    """

    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """ Read group with name as the key from the hdf5 file and return a list of numpy vectors. """
    grp = f[name]
    # 从 h5py.Dataset 对象中获取 NumPy 数组
    return [np.array(grp[str(i)]) for i in range(len(grp))]