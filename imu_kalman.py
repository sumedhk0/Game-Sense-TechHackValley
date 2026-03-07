#!/usr/bin/env python3
"""
IMU Kalman Filter — Real-time 6-DOF orientation estimator
==========================================================
Fuses accelerometer + gyroscope data to estimate:
  - Pitch, Roll  (from accel + gyro, full Kalman)
  - Yaw          (gyro integration only — no magnetometer)

Usage
-----
  # Live simulation (sine-wave IMU)
  python imu_kalman.py

  # Read from CSV  (columns: time,ax,ay,az,gx,gy,gz)
  python imu_kalman.py --input data.csv

  # Read from serial port (Arduino/MPU-6050 @ 115200 baud)
  python imu_kalman.py --port /dev/ttyUSB0 --baud 115200

  # Save filtered output
  python imu_kalman.py --input data.csv --output filtered.csv

  # Disable live plot (headless / server)
  python imu_kalman.py --no-plot

Serial format expected (one line per sample):
  ax,ay,az,gx,gy,gz[,timestamp_ms]
  e.g.  0.12,-0.05,9.81,1.23,-0.45,0.02,12345

Dependencies
------------
  pip install numpy matplotlib pyserial
"""

import argparse
import csv
import math
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Iterator

import numpy as np

# ──────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class IMUSample:
    """One IMU reading."""
    timestamp: float          # seconds
    ax: float                 # accel X  (m/s² or g)
    ay: float                 # accel Y
    az: float                 # accel Z
    gx: float                 # gyro X   (deg/s)
    gy: float                 # gyro Y
    gz: float                 # gyro Z


@dataclass
class FilteredState:
    """Output of the Kalman filter for one sample."""
    timestamp: float
    pitch: float              # degrees
    roll:  float              # degrees
    yaw:   float              # degrees (integrated gyro only)
    pitch_raw: float          # accel-derived (noisy reference)
    roll_raw:  float
    P_pitch: np.ndarray       # 2×2 covariance (pitch axis)
    P_roll:  np.ndarray       # 2×2 covariance (roll axis)
    kalman_gain_pitch: float
    kalman_gain_roll:  float


# ──────────────────────────────────────────────────────────────
#  KALMAN FILTER  (one axis)
# ──────────────────────────────────────────────────────────────

class AxisKalman:
    """
    Single-axis Kalman filter for IMU angle estimation.

    State vector  x = [angle (deg), gyro_bias (deg/s)]

    Predict
    -------
      x̂⁻ = F·x̂   where  F = [[1, -dt], [0, 1]]
      P⁻  = F·P·Fᵀ + Q

    Update
    ------
      y   = z - H·x̂⁻           (innovation, H = [1, 0])
      S   = H·P⁻·Hᵀ + R
      K   = P⁻·Hᵀ / S
      x̂   = x̂⁻ + K·y
      P   = (I - K·H)·P⁻        (Joseph form for stability)
    """

    def __init__(
        self,
        Q_angle:   float = 0.001,   # process noise — angle variance
        Q_bias:    float = 0.003,   # process noise — bias variance
        R_measure: float = 0.03,    # measurement noise variance
    ):
        self.Q_angle   = Q_angle
        self.Q_bias    = Q_bias
        self.R_measure = R_measure

        # State
        self.angle: float = 0.0
        self.bias:  float = 0.0

        # Error covariance  (2×2)
        self.P = np.zeros((2, 2))

        # Last Kalman gain  (scalar, for diagnostics)
        self.K0: float = 0.0

    # ----------------------------------------------------------
    def update(self, accel_angle: float, gyro_rate: float, dt: float) -> float:
        """
        Parameters
        ----------
        accel_angle : angle derived from accelerometer (degrees)
        gyro_rate   : angular rate from gyroscope (degrees/s)
        dt          : elapsed time (seconds)

        Returns
        -------
        Filtered angle estimate (degrees)
        """
        # ---- Predict ----------------------------------------
        rate          = gyro_rate - self.bias
        self.angle   += dt * rate

        # Propagate covariance:  P = F·P·Fᵀ + Q
        self.P[0, 0] += dt * (
            dt * self.P[1, 1]
            - self.P[0, 1]
            - self.P[1, 0]
            + self.Q_angle
        )
        self.P[0, 1] -= dt * self.P[1, 1]
        self.P[1, 0] -= dt * self.P[1, 1]
        self.P[1, 1] += self.Q_bias * dt

        # ---- Update -----------------------------------------
        S      = self.P[0, 0] + self.R_measure     # innovation covariance
        self.K0 = self.P[0, 0] / S                 # Kalman gain (angle row)
        K1     = self.P[1, 0] / S                  # Kalman gain (bias row)

        y             = accel_angle - self.angle    # innovation
        self.angle   += self.K0 * y
        self.bias    += K1      * y

        # Update covariance (Joseph form)
        P00_tmp = self.P[0, 0]
        P01_tmp = self.P[0, 1]
        self.P[0, 0] -= self.K0 * P00_tmp
        self.P[0, 1] -= self.K0 * P01_tmp
        self.P[1, 0] -= K1      * P00_tmp
        self.P[1, 1] -= K1      * P01_tmp

        return self.angle

    # ----------------------------------------------------------
    def reset(self, angle: float = 0.0) -> None:
        self.angle = angle
        self.bias  = 0.0
        self.P     = np.zeros((2, 2))


# ──────────────────────────────────────────────────────────────
#  6-DOF IMU KALMAN WRAPPER
# ──────────────────────────────────────────────────────────────

class IMUKalman:
    """
    Full 6-DOF Kalman filter.

    Pitch & Roll  → AxisKalman (accel + gyro fusion)
    Yaw           → pure gyro integration (no mag reference)
    """

    def __init__(
        self,
        Q_angle:   float = 0.001,
        Q_bias:    float = 0.003,
        R_measure: float = 0.03,
        accel_units: str = "ms2",   # "ms2" | "g"
    ):
        self.pitch_kf = AxisKalman(Q_angle, Q_bias, R_measure)
        self.roll_kf  = AxisKalman(Q_angle, Q_bias, R_measure)

        self.yaw: float   = 0.0
        self.accel_scale  = 1.0 if accel_units == "ms2" else 9.80665

        self._last_ts: Optional[float] = None

    # ----------------------------------------------------------
    def set_params(
        self,
        Q_angle:   Optional[float] = None,
        Q_bias:    Optional[float] = None,
        R_measure: Optional[float] = None,
    ) -> None:
        for kf in (self.pitch_kf, self.roll_kf):
            if Q_angle   is not None: kf.Q_angle   = Q_angle
            if Q_bias    is not None: kf.Q_bias     = Q_bias
            if R_measure is not None: kf.R_measure  = R_measure

    # ----------------------------------------------------------
    def process(self, sample: IMUSample) -> FilteredState:
        """
        Fuse one IMU sample and return the filtered state.
        """
        # Compute dt
        if self._last_ts is None:
            dt = 0.01
        else:
            dt = sample.timestamp - self._last_ts
            dt = max(1e-4, min(dt, 0.5))   # clamp for safety
        self._last_ts = sample.timestamp

        # Scale accelerometer to m/s² if needed
        ax = sample.ax * self.accel_scale
        ay = sample.ay * self.accel_scale
        az = sample.az * self.accel_scale

        # ---- Accel-derived angles (reference, noisy) --------
        # Using atan2 for full ±180° range
        pitch_accel = math.degrees(math.atan2(ay, math.sqrt(ax**2 + az**2)))
        roll_accel  = math.degrees(math.atan2(-ax, az))

        # ---- Kalman update -----------------------------------
        pitch_filt = self.pitch_kf.update(pitch_accel, sample.gy, dt)
        roll_filt  = self.roll_kf.update(roll_accel,  sample.gx, dt)

        # ---- Yaw: gyro integration (complementary) ----------
        self.yaw += sample.gz * dt
        self.yaw  = (self.yaw + 180) % 360 - 180   # wrap to ±180°

        return FilteredState(
            timestamp          = sample.timestamp,
            pitch              = pitch_filt,
            roll               = roll_filt,
            yaw                = self.yaw,
            pitch_raw          = pitch_accel,
            roll_raw           = roll_accel,
            P_pitch            = self.pitch_kf.P.copy(),
            P_roll             = self.roll_kf.P.copy(),
            kalman_gain_pitch  = self.pitch_kf.K0,
            kalman_gain_roll   = self.roll_kf.K0,
        )

    # ----------------------------------------------------------
    def reset(self) -> None:
        self.pitch_kf.reset()
        self.roll_kf.reset()
        self.yaw      = 0.0
        self._last_ts = None


# ──────────────────────────────────────────────────────────────
#  INPUT SOURCES
# ──────────────────────────────────────────────────────────────

def source_simulate(duration: float = 30.0, rate_hz: float = 100.0) -> Iterator[IMUSample]:
    """
    Generate synthetic IMU data: sinusoidal motion + Gaussian noise.
    Useful for testing / demo without hardware.
    """
    g   = 9.80665
    dt  = 1.0 / rate_hz
    t   = 0.0
    rng = np.random.default_rng(42)

    accel_noise = 0.05 * g    # m/s²  (realistic MPU-6050 spec)
    gyro_noise  = 0.5          # deg/s

    while t <= duration:
        # True motion (sinusoidal head movement)
        true_pitch = 20.0 * math.sin(2 * math.pi * 0.3 * t)
        true_roll  = 10.0 * math.sin(2 * math.pi * 0.2 * t + 0.5)
        true_yaw   = 15.0 * math.sin(2 * math.pi * 0.15 * t + 1.0)

        pr = math.radians(true_pitch)
        rr = math.radians(true_roll)

        # Ideal accelerometer reading (gravity projection)
        ax_true =  g * math.sin(rr)
        ay_true = -g * math.sin(pr) * math.cos(rr)
        az_true =  g * math.cos(pr) * math.cos(rr)

        # Ideal gyro reading (time-derivative of euler angles, simplified)
        gx_true = math.degrees(
            math.radians(10.0 * 2*math.pi*0.2 * math.cos(2*math.pi*0.2*t + 0.5))
        )
        gy_true = math.degrees(
            math.radians(20.0 * 2*math.pi*0.3 * math.cos(2*math.pi*0.3*t))
        )
        gz_true = math.degrees(
            math.radians(15.0 * 2*math.pi*0.15 * math.cos(2*math.pi*0.15*t + 1.0))
        )

        yield IMUSample(
            timestamp = t,
            ax = ax_true + rng.normal(0, accel_noise),
            ay = ay_true + rng.normal(0, accel_noise),
            az = az_true + rng.normal(0, accel_noise),
            gx = gx_true + rng.normal(0, gyro_noise),
            gy = gy_true + rng.normal(0, gyro_noise),
            gz = gz_true + rng.normal(0, gyro_noise),
        )

        t  += dt
        time.sleep(0)   # yield control; remove for max-speed batch


def source_csv(path: str) -> Iterator[IMUSample]:
    """
    Read IMU samples from a CSV file.
    Expected header (flexible column order):
      time, ax, ay, az, gx, gy, gz
    OR positional (no header):
      col0=time, col1=ax, col2=ay, col3=az, col4=gx, col5=gy, col6=gz
    """
    with open(path, newline="") as f:
        sample_line = f.readline()
        f.seek(0)
        has_header = not sample_line.strip()[0].lstrip("-").replace(".", "").isdigit()

        reader = csv.reader(f)
        if has_header:
            header = [h.strip().lower() for h in next(reader)]
            idx = {k: header.index(k) for k in ("time","ax","ay","az","gx","gy","gz")}
        else:
            idx = {"time":0,"ax":1,"ay":2,"az":3,"gx":4,"gy":5,"gz":6}

        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                yield IMUSample(
                    timestamp = float(row[idx["time"]]),
                    ax = float(row[idx["ax"]]),
                    ay = float(row[idx["ay"]]),
                    az = float(row[idx["az"]]),
                    gx = float(row[idx["gx"]]),
                    gy = float(row[idx["gy"]]),
                    gz = float(row[idx["gz"]]),
                )
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping malformed row: {row}  ({e})", file=sys.stderr)


def source_serial(port: str, baud: int = 115200) -> Iterator[IMUSample]:
    """
    Read IMU samples from a serial port (e.g. Arduino + MPU-6050).

    Expected serial format per line:
      ax,ay,az,gx,gy,gz[,timestamp_ms]
    Units: accel in m/s² (or g), gyro in deg/s.
    """
    try:
        import serial
    except ImportError:
        sys.exit("[ERROR] pyserial not installed.  Run: pip install pyserial")

    ser = serial.Serial(port, baud, timeout=2)
    print(f"[INFO] Serial port {port} @ {baud} baud opened.", file=sys.stderr)
    t0 = time.time()

    while True:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
            gx, gy, gz = float(parts[3]), float(parts[4]), float(parts[5])
            ts = float(parts[6]) / 1000.0 if len(parts) >= 7 else time.time() - t0
            yield IMUSample(timestamp=ts, ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)
        except ValueError as e:
            print(f"[WARN] Bad serial line: {line!r}  ({e})", file=sys.stderr)


# ──────────────────────────────────────────────────────────────
#  LIVE PLOTTER  (matplotlib, non-blocking)
# ──────────────────────────────────────────────────────────────

class LivePlotter:
    """
    Rolling real-time plot: Raw vs Filtered pitch & roll.
    Runs update() inside the main loop — matplotlib handles its own event loop.
    """
    WINDOW = 500   # samples to display

    def __init__(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        self.plt = plt

        self.fig = plt.figure(figsize=(13, 7), facecolor="#050d14")
        self.fig.canvas.manager.set_window_title("IMU Kalman Filter — Live")

        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.45, wspace=0.3)

        style = dict(facecolor="#030a10")
        self.ax_p  = self.fig.add_subplot(gs[0, :], **style)
        self.ax_r  = self.fig.add_subplot(gs[1, :], **style)
        self.ax_kp = self.fig.add_subplot(gs[2, 0],  **style)
        self.ax_kr = self.fig.add_subplot(gs[2, 1],  **style)

        for ax in (self.ax_p, self.ax_r, self.ax_kp, self.ax_kr):
            ax.tick_params(colors="#3a6a7a", labelsize=7)
            ax.xaxis.label.set_color("#3a6a7a")
            ax.yaxis.label.set_color("#3a6a7a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#0a2030")

        CYAN, RED, GREEN = "#00e5ff", "#ff3d71", "#00ff88"

        # Pitch
        self.ln_p_raw,  = self.ax_p.plot([], [], color=RED,   lw=0.8, alpha=0.6, label="Raw (accel)")
        self.ln_p_filt, = self.ax_p.plot([], [], color=CYAN,  lw=1.2,             label="Filtered (Kalman)")
        self.ax_p.set_title("Pitch (degrees)", color="#7fcfe0", fontsize=9, loc="left")
        self.ax_p.legend(fontsize=7, facecolor="#030a10", edgecolor="#0a2030", labelcolor="w")
        self.ax_p.axhline(0, color="#0a2030", lw=0.5)

        # Roll
        self.ln_r_raw,  = self.ax_r.plot([], [], color=RED,   lw=0.8, alpha=0.6, label="Raw (accel)")
        self.ln_r_filt, = self.ax_r.plot([], [], color=GREEN, lw=1.2,             label="Filtered (Kalman)")
        self.ax_r.set_title("Roll (degrees)", color="#7fcfe0", fontsize=9, loc="left")
        self.ax_r.legend(fontsize=7, facecolor="#030a10", edgecolor="#0a2030", labelcolor="w")
        self.ax_r.axhline(0, color="#0a2030", lw=0.5)

        # Kalman gains
        self.ln_kp, = self.ax_kp.plot([], [], color=CYAN,  lw=1.0)
        self.ln_kr, = self.ax_kr.plot([], [], color=GREEN, lw=1.0)
        self.ax_kp.set_title("Kalman Gain — Pitch", color="#7fcfe0", fontsize=8, loc="left")
        self.ax_kr.set_title("Kalman Gain — Roll",  color="#7fcfe0", fontsize=8, loc="left")

        plt.ion()
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show(block=False)

        # Circular buffers
        N = self.WINDOW
        self._t   = np.zeros(N)
        self._pr  = np.zeros(N)   # pitch raw
        self._pf  = np.zeros(N)   # pitch filtered
        self._rr  = np.zeros(N)   # roll raw
        self._rf  = np.zeros(N)   # roll filtered
        self._kp  = np.zeros(N)
        self._kr  = np.zeros(N)
        self._i   = 0
        self._full = False
        self._render_every = 5
        self._render_count = 0

    def push(self, state: FilteredState) -> None:
        i = self._i % self.WINDOW
        self._t[i]  = state.timestamp
        self._pr[i] = state.pitch_raw
        self._pf[i] = state.pitch
        self._rr[i] = state.roll_raw
        self._rf[i] = state.roll
        self._kp[i] = state.kalman_gain_pitch
        self._kr[i] = state.kalman_gain_roll
        self._i += 1
        if self._i >= self.WINDOW:
            self._full = True

        self._render_count += 1
        if self._render_count >= self._render_every:
            self._render_count = 0
            self._draw()

    def _draw(self) -> None:
        n = self.WINDOW if self._full else self._i
        if n < 2:
            return

        # Build ordered view
        if self._full:
            idx = np.arange(self._i % self.WINDOW, self._i % self.WINDOW + n) % self.WINDOW
        else:
            idx = np.arange(n)

        t  = self._t[idx]
        pr = self._pr[idx]; pf = self._pf[idx]
        rr = self._rr[idx]; rf = self._rf[idx]
        kp = self._kp[idx]; kr = self._kr[idx]

        for ln, data in [
            (self.ln_p_raw, pr), (self.ln_p_filt, pf),
            (self.ln_r_raw, rr), (self.ln_r_filt, rf),
            (self.ln_kp,    kp), (self.ln_kr,     kr),
        ]:
            ln.set_data(t, data)

        for ax in (self.ax_p, self.ax_r, self.ax_kp, self.ax_kr):
            ax.relim(); ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        self.plt.close(self.fig)


# ──────────────────────────────────────────────────────────────
#  OUTPUT WRITER
# ──────────────────────────────────────────────────────────────

class CSVWriter:
    HEADER = [
        "timestamp", "pitch_filt", "roll_filt", "yaw_filt",
        "pitch_raw",  "roll_raw",
        "P_pitch_00", "P_pitch_11",
        "P_roll_00",  "P_roll_11",
        "K_pitch",    "K_roll",
    ]

    def __init__(self, path: str):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def write(self, s: FilteredState) -> None:
        self._w.writerow([
            f"{s.timestamp:.6f}",
            f"{s.pitch:.4f}", f"{s.roll:.4f}", f"{s.yaw:.4f}",
            f"{s.pitch_raw:.4f}", f"{s.roll_raw:.4f}",
            f"{s.P_pitch[0,0]:.6f}", f"{s.P_pitch[1,1]:.6f}",
            f"{s.P_roll[0,0]:.6f}",  f"{s.P_roll[1,1]:.6f}",
            f"{s.kalman_gain_pitch:.6f}", f"{s.kalman_gain_roll:.6f}",
        ])

    def close(self) -> None:
        self._f.close()


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IMU Kalman Filter — fuses accel + gyro to estimate orientation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input",  "-i", metavar="FILE",  help="CSV input file")
    src.add_argument("--port",   "-p", metavar="PORT",  help="Serial port (e.g. /dev/ttyUSB0 or COM3)")

    p.add_argument("--baud",    "-b",  type=int,   default=115200,  help="Serial baud rate")
    p.add_argument("--output",  "-o",  metavar="FILE",              help="Save filtered output to CSV")
    p.add_argument("--no-plot",        action="store_true",         help="Disable live plot")
    p.add_argument("--duration", "-d", type=float, default=30.0,    help="Simulation duration (seconds)")
    p.add_argument("--rate",     "-r", type=float, default=100.0,   help="Simulation sample rate (Hz)")
    p.add_argument("--Q-angle",        type=float, default=0.001,   help="Process noise — angle")
    p.add_argument("--Q-bias",         type=float, default=0.003,   help="Process noise — bias")
    p.add_argument("--R",              type=float, default=0.03,    help="Measurement noise")
    p.add_argument("--accel-units",    choices=["ms2","g"], default="ms2",
                   help="Accelerometer units: ms2 = m/s², g = gravitational units")
    p.add_argument("--quiet",   "-q",  action="store_true",         help="Suppress console output")
    return p


def print_state(s: FilteredState, every: int = 50, _counter: list = [0]) -> None:
    _counter[0] += 1
    if _counter[0] % every != 0:
        return
    print(
        f"t={s.timestamp:8.3f}s | "
        f"Pitch={s.pitch:7.2f}° (raw {s.pitch_raw:7.2f}°)  "
        f"Roll={s.roll:7.2f}° (raw {s.roll_raw:7.2f}°)  "
        f"Yaw={s.yaw:7.2f}°  "
        f"K_p={s.kalman_gain_pitch:.4f}  K_r={s.kalman_gain_roll:.4f}"
    )


def main() -> None:
    args = build_parser().parse_args()

    # ---- Choose input source --------------------------------
    if args.input:
        source = source_csv(args.input)
        print(f"[INFO] Reading from CSV: {args.input}")
    elif args.port:
        source = source_serial(args.port, args.baud)
        print(f"[INFO] Reading from serial: {args.port} @ {args.baud}")
    else:
        print(f"[INFO] Running simulation ({args.duration}s @ {args.rate}Hz)")
        source = source_simulate(args.duration, args.rate)

    # ---- Build filter ---------------------------------------
    kf = IMUKalman(
        Q_angle    = args.Q_angle,
        Q_bias     = args.Q_bias,
        R_measure  = args.R,
        accel_units = args.accel_units,
    )

    # ---- Optional outputs -----------------------------------
    writer  = CSVWriter(args.output) if args.output else None
    plotter = None
    if not args.no_plot:
        try:
            plotter = LivePlotter()
        except Exception as e:
            print(f"[WARN] Could not start live plot: {e}", file=sys.stderr)

    # ---- Main processing loop -------------------------------
    print("[INFO] Processing... (Ctrl+C to stop)\n")
    if not args.quiet:
        print(f"{'t':>10} | {'Pitch_filt':>11} {'Pitch_raw':>10} | "
              f"{'Roll_filt':>10} {'Roll_raw':>9} | {'Yaw':>8} | {'K_p':>8} {'K_r':>8}")
        print("-" * 90)

    n_samples = 0
    t_start   = time.time()

    try:
        for sample in source:
            state = kf.process(sample)

            if not args.quiet:
                print_state(state)

            if writer:
                writer.write(state)

            if plotter:
                plotter.push(state)

            n_samples += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ---- Summary --------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n[DONE] Processed {n_samples:,} samples in {elapsed:.2f}s "
          f"({n_samples/elapsed:.0f} samples/s)")

    if writer:
        writer.close()
        print(f"[INFO] Output saved to: {args.output}")

    if plotter:
        print("[INFO] Close the plot window to exit.")
        try:
            plotter.plt.ioff()
            plotter.plt.show()
        except Exception:
            pass
        plotter.close()


if __name__ == "__main__":
    main()
