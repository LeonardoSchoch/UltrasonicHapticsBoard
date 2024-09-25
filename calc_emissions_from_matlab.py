import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import subprocess
import scipy.io
import os

# Load target image
target_file = 'smiley.png'

# Calculate target amplitude
target_amp = color.rgb2gray(io.imread(target_file))
target_amp = target_amp.astype(np.float64)  # Convert to double
target_amp /= np.max(target_amp)  # Normalize

# Create MATLAB script
matlab_script = f"""
targetfile = 'smiley.png';

targetAmp = double( rgb2gray( imread(targetfile) ) );
targetAmp = targetAmp / max(max(targetAmp)); %normalize

[amps, phases, amp_slice] = calcEmissionForTargetAmpSlice(targetAmp, 0.16, 50, 0.16, 40000, 340, 0.005, 0, 32);

save('results.mat', 'amps', 'phases', 'amp_slice');
"""

# Save temporary MATLAB script
temp_script_path = os.path.join(os.getcwd(), 'temp_script.m').replace('\\', '/')
with open(temp_script_path, 'w') as f:
    f.write(matlab_script)

# Start MATLAB in the background and run the script
matlab_path = r'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'  # Change this to the actual path of your MATLAB installation
subprocess.run([matlab_path, '-batch', f"run('{temp_script_path}')"])

# Load results from the MAT file
results = scipy.io.loadmat('results.mat')
amps = results['amps'].flatten()  # Convert to 1D array
phases = results['phases'].flatten()
amp_slice = results['amp_slice']

# Cleanup: delete temporary files
os.remove(temp_script_path)
os.remove('results.mat')

# Plot target and obtained image
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

# Output mean squared error
target_amp /= np.max(target_amp)  # Normalize
amp_slice /= np.max(amp_slice)  # Normalize
mse = np.sum((target_amp - amp_slice) ** 2) / amp_slice.size
print(mse)
