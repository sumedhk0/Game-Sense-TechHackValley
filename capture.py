import pyaudiowpatch as pyaudio
import numpy as np

import serial
import time

BAUD_RATE = 115200
NUM_MOTORS = 8
SYNC_BYTE = 0xFF
PORT = "COM5"



def _find_loopback(p):
    """Find the WASAPI loopback device matching the default speakers."""
    default_speakers = p.get_default_output_device_info()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["name"].startswith(default_speakers["name"]) and dev.get("isLoopbackDevice", False):
            return dev
    return None


class AudioCapture:
    """Persistent WASAPI loopback audio capture with 30ms windows."""

    def __init__(self):
        self._p = pyaudio.PyAudio()
        loopback = _find_loopback(self._p)
        if not loopback:
            self._p.terminate()
            raise RuntimeError("No WASAPI loopback device found")

        self._rate = int(loopback["defaultSampleRate"])
        self._chunk = int(self._rate * 0.03)  # 30ms window

        self._stream = self._p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self._rate,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=self._chunk,
        )

    def read_levels(self):
        """Read one 30ms chunk and return (left_rms, right_rms)."""
        data = self._stream.read(self._chunk, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)
        left_rms = float(np.sqrt(np.mean(samples[0::2] ** 2)))
        right_rms = float(np.sqrt(np.mean(samples[1::2] ** 2)))
        return left_rms, right_rms

    def close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import time
    from motor_mapping import stereo_to_motors
    with AudioCapture() as cap:
        try:
            ser = serial.Serial(PORT, BAUD_RATE, timeout=1, write_timeout=1)
            print(f"Connected to {PORT} at {BAUD_RATE} baud.")
            time.sleep(2)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            print("Sending wave pattern... Press Ctrl+C to stop.")
            
            while True:
                left_rms, right_rms = cap.read_levels()
                motors = stereo_to_motors(left_rms, right_rms)
                packet = bytearray([SYNC_BYTE]) + bytearray(motors)
                ser.write(packet)
                time.sleep(0.03)
        except serial.SerialException as e:
            print(f"Error opening up serial port: {e}")
        except KeyboardInterrupt:
            print("\nStopping...")
            if 'ser' in locals() and ser.is_open:
                ser.write(bytearray([SYNC_BYTE] + [0]*NUM_MOTORS))
                ser.close()