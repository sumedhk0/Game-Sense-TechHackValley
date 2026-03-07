import serial
import time
import math
import random

# Configuration matching constants.h
BAUD_RATE = 115200
NUM_MOTORS = 8
SYNC_BYTE = 0xFF
PORT = "COM5"


def main():
    val = 0
    increment_up = True
    try:
        # Initialize serial connection
        # timeout=1 ensures we don't hang indefinitely if reading
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1, write_timeout=1)
        print(f"Connected to {PORT} at {BAUD_RATE} baud.")
        
        # Wait for ESP32 to reset after serial connection is established
        time.sleep(2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("Sending wave pattern... Press Ctrl+C to stop.")
        
        while True:
            intensities = []
            if increment_up:
                val += 1
            else:
                val -= 1
            for i in range(NUM_MOTORS):
                intensities.append(val)
            
            print(intensities)
            # Packet structure: [SYNC_BYTE, M1, M2, ..., M8]
            packet = bytearray([SYNC_BYTE]) + bytearray(intensities)
            ser.write(packet)
            
            if val == 254:
                increment_up = False
            elif val == 0:
                increment_up = True
            time.sleep(0.05) # Update at roughly 20Hz

    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")
        if 'ser' in locals() and ser.is_open:
            # Send all zeros to turn off motors before closing
            ser.write(bytearray([SYNC_BYTE] + [0] * NUM_MOTORS))
            ser.close()

if __name__ == "__main__":
    main()
