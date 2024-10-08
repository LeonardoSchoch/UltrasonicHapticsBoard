import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy.io
import subprocess
import os
import serial

matlab_path = 'C:/Program Files/MATLAB/R2024b/bin/matlab.exe'
target_file = 'circle.ply'
com_port = 'COM1'
baud_rate = 9600

grid_size = 32  # max grid size (this size needs to be double the actual grid size of the board, e.g. for a 16x16 board, use 32)
max_dist = 0.25  # max distance from ground plane (in meters)

iterations = 50
board_size = 0.16
emitter_size = 0.01
amp_res = 0
phase_res = 32
emitter_freq = 40000
speed_of_sound = 343

matlab_script = f"""
% Load the data
load('fields.mat');
load('distances.mat');

% Get the number of fields (assumed to be the same as the number of distances)
num_fields = size(fields, 1);
grid_size = size(fields, 2);

% Pre-allocate arrays based on the expected dimensions
amps = zeros(num_fields, {int(board_size / emitter_size)**2});
phases = zeros(num_fields, {int(board_size / emitter_size)**2});
amp_slices = zeros(num_fields, grid_size, grid_size);

% Loop over the number of field entries
for i = 1:num_fields
    % Reshape fields from 1 x grid_size x grid_size to grid_size x grid_size
    field = squeeze(fields(i, :, :));
    
    % Call the function for each field and distance pair
    [temp_amps, temp_phases, temp_amp_slice] = calcEmissionForTargetAmpSlice(field, distances(i), {iterations}, {board_size}, {emitter_freq}, {speed_of_sound}, {emitter_size}, {amp_res}, {phase_res});
    
    % Store the results in the pre-allocated arrays
    amps(i, :) = temp_amps;
    phases(i, :) = temp_phases;
    amp_slices(i, :, :) = temp_amp_slice;
end

% Save the concatenated results
save('results.mat', 'amps', 'phases', 'amp_slices');

"""


def transform_ply_data(filename):
    points = []

    # Read PLY file (binary)
    with open(filename, 'rb') as f:
        # Skip header
        while b'end_header' not in f.readline():
            pass

        # Read 3D points
        while True:
            bytes = f.read(12)  # 3 * 4 Bytes (float)
            if len(bytes) == 12:
                points.append(struct.unpack('fff', bytes))
            else:
                break
            
    points = np.array(points)

    # Get the minimum and maximum values of the x, y and z coordinates
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Normalize the x and y coordinates to [0, grid_size-1] and convert to integers
    coordinates = (points[:, :2] - min_vals[:2]) / (max_vals[:2] - min_vals[:2])  # Normalize to [0, 1]
    coordinates = (coordinates * (grid_size - 1)).astype(int)  # Scale to [0, grid_size-1]

    # Normalize the z coordinates to [0, max_dist]
    distances = points[:, 2] / np.max(points[:, 2]) * max_dist

    # Combine the integer x and y with the z values
    normalized_coords_distances = np.hstack((coordinates, distances.reshape(-1, 1)))
    
    # Number of matrices to create
    num_fields = coordinates.shape[0]

    # Initialize a 3D array to hold all the fields (num_fields, grid_size, grid_size)
    fields = np.zeros((num_fields, grid_size, grid_size))

    # For each coordinate pair, set the corresponding position to 1 in each slice
    for i, (x, y) in enumerate(coordinates):
        fields[i, x, y] = 1
        
    # Save the fields array to a .mat file
    scipy.io.savemat('fields.mat', {'fields': fields})

    # Save the distances array to a .mat file
    scipy.io.savemat('distances.mat', {'distances': distances})
    
    

def run_matlab_script():
    # Save temporary MATLAB script
    temp_script_path = os.path.join(os.getcwd(), 'temp_script.m').replace('\\', '/')
    with open(temp_script_path, 'w') as f:
        f.write(matlab_script)
    
    # Run the MATLAB script
    subprocess.run([matlab_path, '-batch', f"run('{temp_script_path}')"])
    
    # Load results from the .mat file
    results = scipy.io.loadmat('results.mat')
    
    # Clean up temporary files
    os.remove('fields.mat')
    os.remove('distances.mat')
    os.remove('results.mat')
    os.remove(temp_script_path)
    
    return results
    


class PhaseTransmitter:
    
    # Command dictionary to represent the protocol
    commands = {
        'set_phases_amplitudes': 0x80,  # Values below 0x80 are used to set phases or amplitudes
        'start_receiving_phases': 0xFE,
        'swap_buffer': 0xFD
    }

    def __init__(self, port, baudrate=9600, stopbits=serial.STOPBITS_ONE, timeout=1):
        """
        Initialize the serial connection.
        
        Parameters:
        - port: The COM port (e.g., 'COM3', '/dev/ttyUSB0')
        - baudrate: Baud rate for the communication (default 9600)
        - stopbits: Number of stop bits (default 1 stop bit)
        - timeout: Timeout for reading in seconds (default 1 second)
        """
        
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            stopbits=stopbits,
            timeout=timeout
        )
        if self.ser.is_open:
            print(f"Connection to {self.ser.port} established.")
        else:
            print(f"Connection to {self.ser.port} failed.")

    def send_command(self, command):
        """Send a command to the device based on the command protocol."""

        if isinstance(command, (str, int, np.integer)):
            if isinstance(command, str) and command in self.commands:
                command_byte = self.commands[command]
                self.ser.write(command_byte)
                print(f"Sent command: {command}, byte: 0x{command_byte:02X}")
            elif isinstance(command, (int, np.integer)) and command < self.commands['set_phases_amplitudes']:
                self.ser.write(command)
                print(f"Sent command: set_phases_amplitudes, byte: 0x{command:02X}")
            else:
                if isinstance(command, str):
                    print(f"Invalid command: {command}")
                elif isinstance(command, (int, np.integer)):
                    print(f"Invalid command: 0x{command:02X}")
        else:
            print("Command must be a string or integer.")


    def send_phases(self, phases):
        """Send an array of phases to the device, encoded as per the protocol."""
        
        # Send command to start receiving phases
        self.send_command('start_receiving_phases')
        
        for phase in phases:
            # Send each phase as an integer
            self.send_command(phase)
                
        # Once all phases are sent, we can send a command to swap buffers
        self.send_command('swap_buffer')

    def close(self):
        """Close the serial connection."""
        
        self.ser.close()




if __name__ == '__main__':
    
    transform_ply_data(target_file)
    results = run_matlab_script()
    
    phases = results['phases']
    phases = np.round(phases * phase_res / 2).astype(int) % phase_res
    phases[phases < 0] += phase_res
    
    transmitter = PhaseTransmitter(port=com_port, baudrate=baud_rate)

    for i, phase_array in enumerate(phases[:5]):
        print("------------------------------")
        print(f"Sending phase array {i+1}...")
        transmitter.send_phases(phase_array)

    transmitter.close()
    