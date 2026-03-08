import sys
import struct
import numpy as np
import serial
import pyaudiowpatch as pyaudio
import math

NUM_CHANNELS = 8
LFE_INDEX = 3
CHUNK_DURATION_MS = 50
SAMPLE_FORMAT = pyaudio.paFloat32
CHANNEL_NAMES = ["FL", "FR", "FC", "BL", "BR", "SL", "SR"]
MOTOR_ANGLES = [i*45 for i in range(8)]
FIVE_CHANNELS = ['FL','FR','C','SL','SR']

FIVE_CHANNEL_ANGLES = {'FL': 330, 'FR': 30, 'C': 0, 'SL': 250, 'SR': 110}
SERIAL_PORT = "COM5"
BAUD_RATE = 115200
SYNC_BYTE = 0xFF
NUM_MOTORS = 8



def angular_distance(a,b):
    d = abs(a-b) % 360
    return min(d,360-d)

def precompute_weights(falloff=2):
    table = []
    for angle in MOTOR_ANGLES:
        weights = []
        for ch in FIVE_CHANNELS:
            d = angular_distance(angle,FIVE_CHANNEL_ANGLES[ch])
            weights.append(1.0/(1+d**falloff) if d>0 else 1.0)
        total = sum(weights)
        table.append([w /total for w in weights])
    return table

WEIGHT_TABLE = precompute_weights(falloff=2)

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
        if info.get("isLoopbackDevice", False) and info["maxInputChannels"] >= 2:
            return info

    # No suitable device found — print available devices for debugging
    print("No WASAPI loopback device found. Available devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  [{i}] {info['name']} (in={info['maxInputChannels']}, out={info['maxOutputChannels']})")
    sys.exit(1)


def compute_channel_rms(raw_data, num_channels, lfe_index=LFE_INDEX):
    samples = np.frombuffer(raw_data, dtype=np.float32)
    num_frames = len(samples) // num_channels
    if num_frames == 0:
        count = (num_channels-1) if num_channels > 2 else num_channels
        return [0.0] * count
    samples = samples[:num_frames * num_channels]
    channels = samples.reshape(-1, num_channels)
    rms_values = np.sqrt(np.mean(channels ** 2, axis=0))
    if num_channels > 2:
        rms_values = np.delete(rms_values, lfe_index)
    return rms_values.tolist()

#stereo
def two_rms_to_motor_intensities(rms_list):
    l, r = rms_list #CLAUDE: verify that this is correctly outputted.
    motor_values = [
        (l+r)/2,
        0.25*l + 0.75*r,
        r,
        0.0,
        0.0,
        0.0,
        l,
        0.75*l + 0.25*r,
    ]
    return [min(254,max(0,int(v*254))) for v in motor_values]

#5.1
def five_rms_to_motor_intensities(rms_list):
    fl, fr, fc, bl, br = rms_list #CLAUDE: verify that this order is correct and that this is correctly outputted.
    vels = (fl,fr,fc,bl,br)
    motor_values = []
    for weights in WEIGHT_TABLE:
        val = sum(w*v for w,v in zip(weights,vels))
        motor_values.append(min(254, max(0,int(val*254))))
    return motor_values
#7.1
def seven_rms_to_motor_intensities(rms_list):
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
    return [min(254, max(0, int(v * 254) * 10)+ 40) for v in motor_values]



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
        packet = bytearray(NUM_MOTORS + 1)
        packet[0] = SYNC_BYTE
        while True:
            raw_data = stream.read(chunk, exception_on_overflow=False)
            rms_list = compute_channel_rms(raw_data, num_channels=channels)
            output_rms(rms_list)
            if channels == 2:
                intensities = two_rms_to_motor_intensities(rms_list)
            elif channels == 6:
                intensities = five_rms_to_motor_intensities(rms_list)
            elif channels == 8:
                intensities = seven_rms_to_motor_intensities(rms_list)
            else:
                print('Configure speaker setup to use stereo, 5.1, or 7.1');
                sys.exit(1)
            packet[1:] = intensities
            ser.write(packet)

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
