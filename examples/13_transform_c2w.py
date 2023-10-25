''' Examples of transformation & camera model.
'''
import os, sys
import numpy as np
import math
import scipy.io as sio
from skimage import io
from time import time
import subprocess

sys.path.append('..')
import face3d
from face3d import mesh

from ipdb import set_trace as st
import json
from plyfile import PlyData, PlyElement

def transform_test(vertices, obj, camera, h = 256, w = 256, return_c2w = False):
	'''
	Args:
		obj: dict contains obj transform paras
		camera: dict contains camera paras
	'''
	R = mesh.transform.angle2matrix(obj['angles'])
	transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
	
	if camera['proj_type'] == 'orthographic':
		projected_vertices = transformed_vertices
		image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	else:

		## world space to camera space. (Look at camera.) 
		camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		c2w = mesh.transform.lookat_camera_return_c2w(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		## camera space to image space. (Projection) if orth project, omit
		projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
		## to image coords(position in image)
		image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

	rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 1)
	if return_c2w:
		return rendering, c2w
	return rendering

# --------- load mesh data
C = sio.loadmat('Data/example1.mat')
vertices = C['vertices']; 
global colors
global triangles
colors = C['colors']; triangles = C['triangles']
colors = colors/np.max(colors)
# move center to [0,0,0]
vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

# save folder
top_save_folder = 'results/nerf_3dmm'
rel_folder = 'train' # to save images
save_folder = os.path.join(top_save_folder, rel_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
options = '-delay 10 -loop 0 -layers optimize' # gif options. need ImageMagick installed.

# ---- nerf cam json
# Data to store in JSON file
out_data = {
    'camera_angle_x': 30,
}
out_data['frames'] = []

# ---- start
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size

## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'
# scale
for factor in np.arange(0.5, 1.2, 0.1):
	obj['s'] = scale_init*factor
	obj['angles'] = [0, 0, 0]
	obj['t'] = [0, 0, 0]
	image = transform_test(vertices, obj, camera) 
	io.imsave('{}/1_1_{:.2f}.png'.format(save_folder, factor), (np.squeeze(image) * 255).astype(np.uint8))
	
 
#  # Save the rendered image
#     image_filename = '{}/image_{:03d}.png'.format(save_folder, i)
#     io.imsave(image_filename, (np.squeeze(image) * 255).astype(np.uint8))
#     print(f"Saved image: {image_filename}")

# angles
for i in range(3):
	for angle in np.arange(-50, 51, 10):
		obj['s'] = scale_init
		obj['angles'] = [0, 0, 0]
		obj['angles'][i] = angle
		obj['t'] = [0, 0, 0]
		image = transform_test(vertices, obj, camera) 
		io.imsave('{}/1_2_{}_{}.png'.format(save_folder, i, angle), (np.squeeze(image) * 255).astype(np.uint8))
subprocess.call('convert {} {}/1_*.png {}'.format(options, save_folder, save_folder + '/obj.gif'), shell=True)

#### My precious!!!! TODO: 
## TODO-1: get w2c from below, and save as c2w
## TODO-2: once get the c2w, it is now in openGL, convert it to opencv as nerf
## 2. fix obj position(center=[0,0,0], front pose). change camera position&direction, using perspective projection(fovy fixed)
obj['s'] = scale_init
obj['angles'] = [0, 0, 0]
obj['t'] = [0, 0, 0]
# obj: center at [0,0,0]. size:200

### save ply that used for camera rendering
def storePly_gaussian_splatting(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    st()

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    print("Ply data for Gaussian splatting is saved to {} !".format(path))
    exit(0)
    
R = mesh.transform.angle2matrix(obj['angles'])
transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
# storePly(ply_path, xyz, SH2RGB(shs) * 255)
ply_path = '{}/{}'.format(save_folder, 'points3d.ply')
storePly_gaussian_splatting(ply_path, transformed_vertices, C['colors'] * 255)
st()

camera['proj_type'] = 'perspective'
camera['at'] = [0, 0, 0]
camera['near'] = 1000
camera['far'] = -100
# eye position
camera['fovy'] = 30
camera['up'] = [0, 1, 0] #
# z-axis: eye from far to near, looking at the center of face
for p in np.arange(500, 250-1, -10): # 0.5m->0.25m # -40
	camera['eye'] = [0, 0, p]  # stay in front of face~
	# image = transform_test(vertices, obj, camera) 
	image, c2w = transform_test(vertices, obj, camera, return_c2w = True) 
	# print(c2w)
	filepath = '{}/2_eye_1_{}.png'.format(save_folder, 1000-p)
	io.imsave(filepath, (np.squeeze(image) * 255).astype(np.uint8))
	_basename, _ext = os.path.splitext(os.path.basename(filepath))
	frame_data = {
			'file_path': os.path.join(rel_folder,_basename),
			# 'rotation': radians(stepsize),
			'transform_matrix': c2w.tolist()
		}
	print(frame_data)
	out_data['frames'].append(frame_data)

# y-axis: eye from down to up, looking at the center of face
for p in np.arange(-300, 301, 10): # up 0.3m -> down 0.3m #60
	camera['eye'] = [0, p, 250] # stay 0.25m far
	# image = transform_test(vertices, obj, camera) 
	image, c2w = transform_test(vertices, obj, camera, return_c2w = True) 
	# print(c2w)
	filepath = '{}/2_eye_2_{}.png'.format(save_folder, p/6)
	io.imsave(filepath, (np.squeeze(image) * 255).astype(np.uint8))
	_basename, _ext = os.path.splitext(os.path.basename(filepath))
	frame_data = {
			'file_path': os.path.join(rel_folder,_basename),
			# 'rotation': radians(stepsize),
			'transform_matrix': c2w.tolist()
		}
	# print(frame_data)
	out_data['frames'].append(frame_data)

# x-axis: eye from left to right, looking at the center of face
for p in np.arange(-300, 301, 10): # left 0.3m -> right 0.3m #60
	camera['eye'] = [p, 0, 250] # stay 0.25m far
	# image = transform_test(vertices, obj, camera) 
	image, c2w = transform_test(vertices, obj, camera, return_c2w = True)
	# print(c2w)
	filepath = '{}/2_eye_3_{}.png'.format(save_folder, -p/6)
	io.imsave(filepath, (np.squeeze(image) * 255).astype(np.uint8))
	_basename, _ext = os.path.splitext(os.path.basename(filepath))
	frame_data = {
			'file_path': os.path.join(rel_folder,_basename),
			# 'rotation': radians(stepsize),
			'transform_matrix': c2w.tolist()
		}
	# print(frame_data)
	out_data['frames'].append(frame_data)

# up direction
camera['eye'] = [0, 0, 250] # stay in front
for p in np.arange(-50, 51, 5): #10
	world_up = np.array([0, 1, 0]) # default direction
	z = np.deg2rad(p)
	Rz=np.array([[math.cos(z), -math.sin(z), 0],
                 [math.sin(z),  math.cos(z), 0],
                 [     0,       0, 1]])
	up = Rz.dot(world_up[:, np.newaxis]) # rotate up direction
	# note that: rotating up direction is opposite to rotating obj
	# just imagine: rotating camera 20 degree clockwise, is equal to keeping camera fixed and rotating obj 20 degree anticlockwise.
	camera['up'] = np.squeeze(up)
	# image = transform_test(vertices, obj, camera) 
	image, c2w = transform_test(vertices, obj, camera, return_c2w = True) 
	# print(c2w)
	filepath = '{}/2_eye_4_{}.png'.format(save_folder, -p)
	io.imsave(filepath, (np.squeeze(image) * 255).astype(np.uint8))
	_basename, _ext = os.path.splitext(os.path.basename(filepath))
	frame_data = {
			'file_path': os.path.join(rel_folder,_basename),
			# 'rotation': radians(stepsize),
			'transform_matrix': c2w.tolist()
		}
	# print(frame_data)
	out_data['frames'].append(frame_data)

# st()

def save_gif(morphed_image, fname):
  res_list = []
  k = 0
  while k < morphed_image.shape[0]:
    res_list.append(morphed_image[k, :, :, :].astype(np.uint8))
    k += 1
  imageio.mimsave(fname, res_list)
  
subprocess.call('convert {} {}/2_*.png {}'.format(options, save_folder, save_folder + '/camera.gif'), shell=True)

# -- delete png files
print('gifs have been generated')
# subprocess.call('rm {}/*.png'.format(save_folder), shell=True)

## TODO: save transformation.json as nerf
with open(top_save_folder + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)	