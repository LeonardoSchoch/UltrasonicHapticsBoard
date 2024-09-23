import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from calcEmissionForTargetAmpSlice import calc_emission_for_target_amp_slice

# Zielbild laden
target_file = 'smiley.png'

# Zielamplitude berechnen
target_amp = color.rgb2gray(io.imread(target_file))
target_amp = target_amp.astype(np.float64)  # Umwandlung in double
target_amp /= np.max(target_amp)  # Normalisieren

# Emission berechnen
amps, phases, amp_slice = calc_emission_for_target_amp_slice(target_amp, 0.16, 50, 0.16, 40000, 340, 0.005, 0, 32)

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
