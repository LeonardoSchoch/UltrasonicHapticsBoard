import numpy as np
import struct

def generate_ply_sphere(filename, radius=1, num_theta=50, num_phi=25):
    # Generate the sphere point cloud with evenly spaced Theta and Phi values
    theta = np.linspace(0, 2 * np.pi, num_theta)
    phi = np.linspace(0, np.pi, num_phi)

    # Create a meshgrid of Theta and Phi (2D arrays)
    theta, phi = np.meshgrid(theta, phi)

    # Calculate the x, y, z coordinates of the sphere using vectorized computation
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Flatten the arrays for export
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    num_points = points.shape[0]

    # Write the binary PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
    
    # Write the points to a binary PLY file
    with open(filename, 'wb') as f:
        f.write(header.encode('utf-8'))
        for point in points:
            f.write(struct.pack('fff', *point))

# Save the sphere as PLY file
generate_ply_sphere("sphere.ply")
