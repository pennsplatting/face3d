''' 
Generate 2d maps representing different attributes(colors, depth, pncc, etc)
: render attributes to image space.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh
import plyfile
from plyfile import PlyData, PlyElement

# load vertices
ply_path = '/home/xuyimeng/Repo/face3d/examples/results/image_map/uv_vertices_700norm_final.ply'
plydata = PlyData.read(ply_path)
vertices = np.array(plydata['vertex'].data)


## use uv coordinates to get texture
uv_map = io.imread('results/uv_map/uv_texture_map_512.jpg')
uv_h, uv_w = uv_map.shape[:2]

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords
uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/Out/BFM_UV.mat') #(53215, 2)
uv_coords_processed = process_uv(uv_coords, uv_h, uv_w)
uv_coords_processed = uv_coords_processed.astype(np.int32)
uv_texture = uv_map[uv_coords_processed[:,1],uv_coords_processed[:,0]]
