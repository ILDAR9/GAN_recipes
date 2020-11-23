import scipy.io as io
import scipy.ndimage as nd
import numpy as np
import os
import scipy.io as spio
import matplotlib.pyplot as plt



data_fld = "3DShapeNets/volumetric_data"
# load .mat file
voxels = io.loadmat(os.path.join(data_fld, "car/30/train/car_000000010_2.mat"))['instance']
# print(sorted(voxels.keys()))
print(voxels.shape)
# 30 -> 32
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
print(voxels.shape)
# 32 -> 64
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
print(voxels.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('auto')
ax.voxels(voxels, edgecolor="red")
plt.show()
# plt.savefig(file_path)