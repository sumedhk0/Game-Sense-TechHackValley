import sys
import struct
import numpy as np
import serial
import pyaudiowpatch as pyaudio

NUM_CHANNELS = 8
LFE_INDEX = 3
CHUNK_DURATION_MS = 50
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNEL_NAMES = ["FL", "FR", "FC", "BL", "BR", "SL", "SR"]

SERIAL_PORT = "COM5"
BAUD_RATE = 115200
SYNC_BYTE = 0xFF
NUM_MOTORS = 8


def get_loopback_device(p):
    try:
        device_info = p.get_default_wasapi_loopback()
        if device_info["maxInputChannels"] >= NUM_CHANNELS:
            return device_info
        print(f"Default loopback has {device_info['maxInputChannels']} channels, need {NUM_CHANNELS}.")
    except OSError:
        pass

    # Fallback: search all devices
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("isLoopbackDevice", False) and info["maxInputChannels"] >= NUM_CHANNELS:
            return info

    # No suitable device found — print available devices for debugging
    print("No 7.1 WASAPI loopback device found. Available devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  [{i}] {info['name']} (in={info['maxInputChannels']}, out={info['maxOutputChannels']})")
    sys.exit(1)


def compute_channel_rms(raw_data, num_channels=NUM_CHANNELS, lfe_index=LFE_INDEX):
    samples = np.frombuffer(raw_data, dtype=np.float32)
    num_frames = len(samples) // num_channels
    if num_frames == 0:
        return [0.0] * (num_channels - 1)
    samples = samples[:num_frames * num_channels]
    channels = samples.reshape(-1, num_channels)
    rms_values = np.sqrt(np.mean(channels ** 2, axis=0))
    rms_values = np.delete(rms_values, lfe_index)
    return rms_values.tolist()


def rms_to_motor_intensities(rms_list):
    """Map 7 audio channel RMS values to 8 motor intensities (0-255).

    RMS list indices: FL=0, FR=1, FC=2, BL=3, BR=4, SL=5, SR=6
    Motor order: Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left
    """
    fl, fr, fc, bl, br, sl, sr = rms_list
    motor_values = [
        fc,              # 0: Front
        fr,              # 1: Front-Right
        sr,              # 2: Right
        br,              # 3: Back-Right
        (bl + br) / 2,   # 4: Back
        bl,              # 5: Back-Left
        sl,              # 6: Left
        fl,              # 7: Front-Left
    ]
    return [min(255, max(0, int(v * 255))) for v in motor_values]


def send_serial(ser, intensities):
    """Send sync byte + 8 motor intensity bytes over serial."""
    packet = struct.pack("B" * 9, SYNC_BYTE, *intensities)
    ser.write(packet)


def output_rms(rms_list):
    formatted = [f"{v:.4f}" for v in rms_list]
    print(f"[{', '.join(formatted)}]")


def main():
    p = pyaudio.PyAudio()
    ser = None
    stream = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        print(f"Serial: {SERIAL_PORT} @ {BAUD_RATE} baud")

        device_info = get_loopback_device(p)
        sample_rate = int(device_info["defaultSampleRate"])
        channels = device_info["maxInputChannels"]
        chunk = int(sample_rate * CHUNK_DURATION_MS / 1000)

        print(f"Device: {device_info['name']}")
        print(f"Sample rate: {sample_rate} Hz | Channels: {channels} | Chunk: {chunk} frames")
        print(f"Output channels: {CHANNEL_NAMES}")

        stream = p.open(
            format=SAMPLE_FORMAT,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=chunk,
        )

        print("Capturing... Press Ctrl+C to stop.\n")
        while True:
            raw_data = stream.read(chunk, exception_on_overflow=False)
            rms_list = compute_channel_rms(raw_data, num_channels=channels)
            output_rms(rms_list)
            intensities = rms_to_motor_intensities(rms_list)
            send_serial(ser, intensities)

    except KeyboardInterrupt:
        print("\nStopping capture.")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if ser is not None:
            ser.close()
        p.terminate()


if __name__ == "__main__":
    main()
