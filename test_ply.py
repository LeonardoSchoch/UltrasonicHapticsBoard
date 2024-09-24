import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def write_binary_ply_sphere(filename, radius=1, num_theta=50, num_phi=25):
    # Erzeuge die Kugelpunktwolke mit gleichmäßig verteilten Theta- und Phi-Werten
    theta = np.linspace(0, 2 * np.pi, num_theta)
    phi = np.linspace(0, np.pi, num_phi)

    # Erzeuge ein Meshgrid von Theta und Phi (2D Arrays)
    theta, phi = np.meshgrid(theta, phi)

    # Berechne die x, y, z Koordinaten der Kugel mit vektorisierter Berechnung
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Flachstellen der Arrays für den Export
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    num_points = points.shape[0]

    # Schreibe das binäre PLY-Header
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
    
    # Schreibe die Punkte in eine binäre PLY-Datei
    with open(filename, 'wb') as f:
        f.write(header.encode('utf-8'))
        for point in points:
            f.write(struct.pack('fff', *point))

# Speichern der Kugel als PLY-Datei
write_binary_ply_sphere("sphere.ply")



def plot_sphere_ply(filename):
    points = []

    # PLY-Datei auslesen (binär)
    with open(filename, 'rb') as f:
        # Header überspringen
        while b'end_header' not in f.readline():
            pass

        # 3D Punkte auslesen
        while True:
            bytes = f.read(12)  # 3 * 4 Bytes (float)
            if len(bytes) == 12:
                points.append(struct.unpack('fff', bytes))
            else:
                break

    points = np.array(points)
    
    # 3D Plot erstellen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Kugelpunktwolke plotten
plot_sphere_ply("sphere.ply")
