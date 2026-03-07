#ifndef BLE_KEYBOARD_HID_H
#define BLE_KEYBOARD_HID_H

// Call once in setup(). Starts BLE advertising as "GameSense Controller".
void bleKeyboardInit();

// Returns true if a device is currently connected via BLE.
bool bleKeyboardIsConnected();

// Call each loop iteration with current tilt angles (degrees).
//   pitch > 0 → forward tilt → W    pitch < 0 → backward tilt → S
//   roll  < 0 → left tilt    → A    roll  > 0 → right tilt    → D
// Handles key press/release with hysteresis internally.
void bleKeyboardUpdateFromTilt(float pitch, float roll);

#endif
