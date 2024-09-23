import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import zoom

def calc_emission_for_target_amp_slice(target_amp_slice, dist, iters, slice_size, freq, sound_speed, emitter_size, amp_res, phase_res):
    """
    Berechnet die Emission für eine gegebene Zielamplitudenmatrix.

    Argumente:
    - target_amp_slice: Matrix der Größe [nx, ny], die die Zielamplitude enthält (soll quadratisch sein und eine Breite haben, die eine Zweierpotenz ist).
    - dist: Abstand von der Emissionsscheibe zur Zielscheibe (in Metern).
    - iters: Anzahl der Iterationen des Algorithmus.
    - slice_size: Seitenlänge der Emissions- und Zielscheibe (in Metern).
    - freq: Frequenz der Wellen (in Hz).
    - sound_speed: Ausbreitungsgeschwindigkeit des Schalls (in m/s).
    - emitter_size: Durchmesser der Emitter (in Metern).
    - amp_res: Amplitudenauflösung (0 = keine Amplitudensteuerung).
    - phase_res: Phasenauflösung.

    Rückgabe:
    - amps: Array der Größe [1, n_emitters], das die Amplituden für jeden Emitter enthält.
    - phases: Array der Größe [1, n_emitters], das die Phasen für jeden Emitter enthält.
    - amp_slice: Matrix der Größe [nx, ny], die die berechnete Amplitudenverteilung an der Zielscheibe enthält.
    """

    w, h = target_amp_slice.shape
    assert w == h
    assert 2 ** np.ceil(np.log2(w)) == w

    target = np.zeros((w, h), dtype=complex)
    emission = np.zeros((w, h), dtype=complex)
    n_emitters_per_side = int(np.floor(slice_size / emitter_size))
    n_emitters = n_emitters_per_side * n_emitters_per_side
    emitter_px = w / n_emitters_per_side

    # Amplitudenmaske: ein Gitter von Kreisen, die die Emitter repräsentieren
    mask = np.zeros((w, h))
    em_px2 = (emitter_px * emitter_px) / 4
    for ix in range(w):
        for iy in range(h):
            diff_x = ix - (int(ix / emitter_px) * emitter_px + emitter_px / 2)
            diff_y = iy - (int(iy / emitter_px) * emitter_px + emitter_px / 2)
            if diff_x ** 2 + diff_y ** 2 < em_px2:
                mask[ix, iy] = 1

    medium = {'soundspeed': sound_speed, 'attenuationdBcmMHz': 0}  # Für Luft könnten wir 1.61 verwenden

    for ii in range(iters):
        # Stempel target_amp_slice in das Ziel ein. Die Phasen bleiben erhalten.
        target = target_amp_slice * np.exp(1j * np.angle(target))

        # Rückwärtspropagierung der Zielscheibe zur Emissionsscheibe
        emission = fftasa(target, -dist, medium, w, slice_size / w, freq)

        # Einschränkungen auf der Emissionsscheibe anwenden
        amp = np.abs(emission)
        phase = np.angle(emission)

        # Herunterskalieren, um die von jedem Emitter belegten Pixel auf einen Pixel zu mitteln
        down_amp = zoom(amp, (n_emitters_per_side / w, n_emitters_per_side / h), order=1)
        down_phase = zoom(phase, (n_emitters_per_side / w, n_emitters_per_side / h), order=1)

        # Amplitude und Phase diskretisieren
        if amp_res == 0:
            down_amp = np.ones_like(down_amp)  # Alle auf Eins setzen
        else:
            down_amp = np.floor(down_amp * amp_res) / amp_res

        if phase_res == 0:
            down_phase = np.zeros_like(down_phase)  # Alle auf Null setzen
        else:
            down_phase = np.floor(down_phase / np.pi * phase_res) * np.pi / phase_res

        # Hochskalieren mit "nearest", damit die Amplitude/Phase in allen Pixeln des Emitters gleich ist
        amp = zoom(down_amp, (w / n_emitters_per_side, h / n_emitters_per_side), order=0)
        phase = zoom(down_phase, (w / n_emitters_per_side, h / n_emitters_per_side), order=0)

        # Maske auf Emission anwenden, d.h. die kreisförmige Form der Emitter
        amp = amp * mask

        # Amplitude und Phase in die komplexe Darstellung umwandeln
        emission = amp * np.exp(1j * phase)

        # Propagierung der Emission zur Zielscheibe
        target = fftasa(emission, dist, medium, w, slice_size / w, freq)

    amp_slice = np.abs(target)

    # Amplituden und Phasen für jeden Emitter extrahieren
    amps = np.zeros(n_emitters)
    phases = np.zeros(n_emitters)
    index = 0
    for ix in range(n_emitters_per_side):
        for iy in range(n_emitters_per_side):
            center_x = int(round(ix * emitter_px - emitter_px / 2))
            center_y = int(round(iy * emitter_px - emitter_px / 2))
            # Holen der Emissionsamplitude/phase im Zentrum des Emitters
            em = emission[center_x, center_y]
            amps[index] = np.abs(em)
            phases[index] = np.angle(em)
            index += 1

    return amps, phases, amp_slice



def fftasa(p0, z, medium, N, delta, f0):
    """
    Berechnet die Druckverteilung in einem Wellenfeld basierend auf dem Spektralpropagator und der Winkel-Spektrum-Theorie.
    
    Argumente:
    - p0: Matrix der Größe [nx, ny], die den Eingangsdruck (komplex) enthält
    - z: Position der Ebene (in Metern)
    - medium: Dictionary mit den Eigenschaften des Mediums (Schallgeschwindigkeit und Dämpfung)
    - N: Anzahl der Gitterpunkte für die FFT
    - delta: Räumliche Abtastung (in Metern)
    - f0: Frequenz (in Hz)

    Rückgabe:
    - fftpress: Matrix der Größe [nx, ny], die den berechneten Druck enthält.
    """

    # Wellenlänge berechnen
    wavelen = medium['soundspeed'] / f0
    dB_per_neper = 20 * np.log10(np.exp(1))
    attenuation_nepers_per_meter = (medium['attenuationdBcmMHz'] / dB_per_neper) * 100 * f0 / 1e6

    # Fourier-Transformation der Eingangsdruckverteilung
    fftpressz0 = np.fft.fft2(p0, [N, N])

    # Wellenzahl berechnen
    wavenum = 2 * np.pi / wavelen
    if N % 2:  # Ungerade Anzahl
        kx = np.linspace(-N/2 - 0.5, N/2 - 1.5, N) * wavelen / (N * delta)
        ky = np.linspace(-N/2 - 0.5, N/2 - 1.5, N) * wavelen / (N * delta)
    else:  # Gerade Anzahl
        kx = np.linspace(-N/2, N/2 - 1, N) * wavelen / (N * delta)
        ky = np.linspace(-N/2, N/2 - 1, N) * wavelen / (N * delta)

    # Gitter für kx und ky
    kxspace, kyspace = np.meshgrid(kx, ky)
    kxsq_ysq = np.fft.fftshift(kxspace**2 + kyspace**2)
    kzspace = wavenum * np.sqrt(1 - kxsq_ysq)

    # Grundlegender Spektralpropagator
    if z > 0:
        H = np.conj(np.exp(1j * z * kzspace))
    else:
        H = np.exp(-1j * z * kzspace) * (kxsq_ysq <= 1)

    # Dämpfung berücksichtigen
    if attenuation_nepers_per_meter > 0:
        evanescent_mode = np.sqrt(kxsq_ysq) < 1
        H = H * np.exp(-attenuation_nepers_per_meter * z / np.cos(np.arcsin(np.sqrt(kxsq_ysq)))) * evanescent_mode

    # Winkelbegrenzung
    D = (N - 1) * delta
    thres = np.sqrt(0.5 * D**2 / (0.5 * D**2 + z**2))
    filt = (np.sqrt(kxsq_ysq) <= thres)
    H = H * filt

    # Rücktransformation in den Raum
    fftpress = np.fft.ifft2(fftpressz0 * H, [N, N])

    return fftpress

