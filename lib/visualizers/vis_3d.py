import cv2 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def vis_3d(v, ray_o, ray_d, size):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(ray_o.shape[1]):
        x = v[0, i, :, 0]
        y = v[0, i, :, 1]
        z = v[0, i, :, 2]
        rx = ray_o[0, i, 0]
        ry = ray_o[0, i, 1]
        rz = ray_o[0, i, 2]
        ax.scatter(x,y,z, s=size)
        ax.scatter(rx, ry, rz, s = 20, c = 'Green')
    plt.show()