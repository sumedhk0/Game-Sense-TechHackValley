import pyaudiowpatch as pyaudio
import numpy as np


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


def get_stereo_levels():
    """One-shot capture (opens and closes stream). Kept for backward compat."""
    with AudioCapture() as cap:
        l, r = cap.read_levels()
    return [l, r]


if __name__ == "__main__":
    import time
    from motor_mapping import stereo_to_motors

    MOTOR_LABELS = ["L", "FL", "F", "FR", "R", "BR", "B", "BL"]

    with AudioCapture() as cap:
        print("Monitoring stereo levels + motor vector (Ctrl+C to stop)...")
        try:
            while True:
                left_rms, right_rms = cap.read_levels()
                motors = stereo_to_motors(left_rms, right_rms)

                bar_l = "█" * int(left_rms * 50)
                bar_r = "█" * int(right_rms * 50)
                motor_str = " ".join(f"{lbl}:{v:3d}" for lbl, v in zip(MOTOR_LABELS, motors))
                print(f"\rL:{left_rms:.3f} {bar_l:<20} R:{right_rms:.3f} {bar_r:<20} | {motor_str}", end="")

                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
    print()
