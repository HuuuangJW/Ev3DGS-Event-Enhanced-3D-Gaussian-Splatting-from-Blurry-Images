import numpy as np
import math

path = "process/input/letter/poses_bounds.npy"
pose_array = np.load(path)
poses = pose_array[:, :-2].reshape(150,3,5)
# print(poses)
hwf = poses[0,:3,-1]
H, W, focal = hwf
poses = poses[:,:3,:4]
print(H,W,focal)


poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
# print(poses[0])
m = np.array([0,0,0,1])
c2w = poses[0]
c2w = np.insert(c2w, 3, m, axis=0)

c2w[:3, 1:3] *= -1
w2c = np.linalg.inv(c2w)
print(c2w)
R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
T = w2c[:3, 3]
print(c2w)

print(R)
print(T)
fovx = 2*math.atan(W/(2*focal))
fovy = 2*math.atan(H/(2*focal))
print(fovx)
print(fovy)
