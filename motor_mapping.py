import math

# Motor angles in radians (clockwise from left)
# Motor 0: Left (π), Motor 1: Front-Left (3π/4), Motor 2: Front (π/2),
# Motor 3: Front-Right (π/4), Motor 4: Right (0)
FRONT_MOTOR_ANGLES = [math.pi, 3 * math.pi / 4, math.pi / 2, math.pi / 4, 0.0]

# Volume gain: RMS float32 audio is typically 0.0–1.0, but usually well below 1.0.
# This gain factor scales the RMS average up to fill the 0–254 range.
VOLUME_GAIN = 4.0


def stereo_to_motors(left_rms: float, right_rms: float) -> list[int]:
    """Convert stereo L/R RMS levels to an 8-motor intensity vector [0–254].

    Motors are arranged in a circle at 45° intervals, starting at left (π)
    and incrementing clockwise. Back motors mirror their front counterparts.
    """
    total = left_rms + right_rms

    # Silence — all motors off
    if total < 1e-8:
        return [0] * 8

    # Balance: -1 = full left, 0 = center, +1 = full right
    balance = (right_rms - left_rms) / total

    # Map balance to angle on front semicircle
    # balance -1 → π (left), 0 → π/2 (front), +1 → 0 (right)
    sound_angle = math.pi / 2.0 * (1.0 - balance)

    # Overall volume scaled to [0, 254]
    volume = min(total / 2.0 * VOLUME_GAIN * 254.0, 254.0)

    # Compute front semicircle motor intensities (motors 0–4)
    front = []
    for angle in FRONT_MOTOR_ANGLES:
        delta = abs(sound_angle - angle)
        intensity = max(0.0, math.cos(delta)) * volume
        front.append(int(min(intensity, 254.0)))

    # Build full 8-motor vector with back mirroring
    # Motor 5 (back-right) = Motor 3 (front-right)
    # Motor 6 (back)       = Motor 2 (front)
    # Motor 7 (back-left)  = Motor 1 (front-left)
    
    return [front[0], front[1], front[2], front[3], front[4],
            front[3], front[2], front[1]]
