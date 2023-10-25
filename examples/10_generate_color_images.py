''' 
Generate 2d uv maps representing different attributes(colors, depth, image position, etc)
: render attributes to uv space.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

from ipdb import set_trace as st


# # --load mesh data
# C = sio.loadmat('Data/example1.mat')
# vertices = C['vertices']; colors = C['colors']; triangles = C['full_triangles']; 
# colors = colors/np.max(colors)
# # --modify vertices(transformation. change position of obj)
# s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
# R = mesh.transform.angle2matrix([-10, -35, 20]) 
# t = [0, 0, 0]
# transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

# # -- start
# save_folder = 'results/gaussian_splatting'
# if not os.path.exists(save_folder):
#     os.mkdir(save_folder)

# image_h = image_w = 256

# projected_vertices = transformed_vertices.copy() # use standard camera & orth projection here
# image_vertices = mesh.transform.to_image(projected_vertices, image_h, image_w) 
# image = mesh.render.render_colors(image_vertices, triangles, colors, image_h, image_w, c=3)
# io.imsave('{}/image.jpg'.format(save_folder), (np.squeeze(image)* 255).astype(np.uint8))

# Load mesh data
C = sio.loadmat('Data/example1.mat')
vertices = C['vertices']
colors = C['colors']
triangles = C['full_triangles']
colors = colors / np.max(colors)

# Set the image size
image_h = image_w = 256


# Define the number of images you want to render
num_images = 280

# Initialize initial rotation
current_rotation = [0, 0, 0]

# Define rotation sequences to form a square
max_angle = 15

# Create a directory to save the rendered images
save_folder = 'results/gaussian_splatting/{}_{}d'.format(num_images, max_angle)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
    
rotation_sequences = np.asarray([
    [0,0,0], #dummy
    [-max_angle, 0, 0], # Rotate to -x direction
    [0, -max_angle, 0],  # Rotate to -y direction
    [max_angle*2, 0, 0],   # Rotate to y direction
    [0, max_angle*2, 0], # Rotate to -x direction again
    [-max_angle*2, 0, 0],  # Rotate to -y direction again
    [0, -max_angle, 0],  # Rotate to -y direction again
    [max_angle, 0, 0], # Rotate to -x direction
])

# Determine the number of frames for each rotation
frames_per_sequence = num_images // (len(rotation_sequences)-1)

for i in range(num_images):
    # Calculate the current sequence and frame within the sequence
    sequence_index = i // frames_per_sequence + 1
    frame_within_sequence = i % frames_per_sequence
    
    # Set the current rotation based on the sequence and frame
    incremental_rotation = [angle / frames_per_sequence * frame_within_sequence * 1.0 for angle in rotation_sequences[sequence_index]]
    accumulated_rotation = np.stack([rotation_sequences[prev_sequence_index] for prev_sequence_index in range(sequence_index)], axis=0).sum(0)
    # print(incremental_rotation, accumulated_rotation)
    current_rotation = incremental_rotation + accumulated_rotation
    # print(current_rotation)
    # continue

    # Modify vertices (transformation - change position of obj)
    s = 180 / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    
    R = mesh.transform.angle2matrix(current_rotation)
    t = [0, 0, 0]
    
    transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

    # Use standard camera & orthographic projection here
    projected_vertices = transformed_vertices.copy()

    # Transform vertices to image coordinates
    image_vertices = mesh.transform.to_image(projected_vertices, image_h, image_w)

    # Render the image
    image = mesh.render.render_colors(image_vertices, triangles, colors, image_h, image_w, c=3)

    # Save the rendered image
    image_filename = '{}/image_{:03d}.jpg'.format(save_folder, i)
    io.imsave(image_filename, (np.squeeze(image) * 255).astype(np.uint8))
    print(f"Saved image: {image_filename}")
