import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import subprocess
import scipy.io
import os

# Zielbild laden
target_file = 'smiley.png'

# Zielamplitude berechnen
target_amp = color.rgb2gray(io.imread(target_file))
target_amp = target_amp.astype(np.float64)  # Umwandlung in double
target_amp /= np.max(target_amp)  # Normalisieren

# MATLAB-Skript erstellen
matlab_script = f"""
targetfile = 'smiley.png';

targetAmp = double( rgb2gray( imread(targetfile) ) );
targetAmp = targetAmp / max(max(targetAmp)); %normalize

[amps, phases, amp_slice] = calcEmissionForTargetAmpSlice(targetAmp, 0.16, 50, 0.16, 40000, 340, 0.005, 0, 32);

save('results.mat', 'amps', 'phases', 'amp_slice');
"""

# Temporäres MATLAB-Skript speichern
temp_script_path = os.path.join(os.getcwd(), 'temp_script.m').replace('\\', '/')
with open(temp_script_path, 'w') as f:
    f.write(matlab_script)

# MATLAB im Hintergrund starten und das Skript ausführen
matlab_path = r'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'  # Ändere das auf den tatsächlichen Pfad zu deiner MATLAB-Installation
subprocess.run([matlab_path, '-batch', f"run('{temp_script_path}')"])


# Ergebnisse aus der MAT-Datei laden
results = scipy.io.loadmat('results.mat')
amps = results['amps'].flatten()  # Umwandeln in ein 1D-Array
phases = results['phases'].flatten()
amp_slice = results['amp_slice']

# Bereinigung: temporäre Dateien löschen
os.remove(temp_script_path)
os.remove('results.mat')

# Ziel und erhaltenes Bild plotten
plt.subplot(2, 1, 1)
plt.imshow(target_amp, cmap='gray')
plt.title('Target')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(amp_slice, cmap='gray')
plt.title('Obtained')
plt.axis('off')

plt.tight_layout()
plt.show()

# Mittlere quadratische Abweichung ausgeben
target_amp /= np.max(target_amp)  # Normalisieren
amp_slice /= np.max(amp_slice)  # Normalisieren
mse = np.sum((target_amp - amp_slice) ** 2) / amp_slice.size
print(mse)
