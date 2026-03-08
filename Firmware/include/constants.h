#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>
#include <array>

constexpr uint32_t BAUD_RATE = 115200;
constexpr uint8_t  NUM_MOTORS = 8;

constexpr uint32_t PWM_FREQUENCY  = 1000;
constexpr uint8_t  PWM_RESOLUTION = 8;

constexpr uint8_t CS_PIN = 2;
constexpr uint8_t SCK_PIN = 18;
constexpr uint8_t MISO_PIN = 19;
constexpr uint8_t MOSI_PIN = 23;

constexpr uint8_t SWITCH_PIN = 35;

// Pin assignments: Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left
constexpr std::array<uint8_t, NUM_MOTORS> MOTOR_PINS = {27, 21, 22, 33, 32, 25, 26, 13};
/* Pins:
    25 - back left
    32 - back?
    22 - Right
    26 - left
    27 - Center CORRECT
    21 - Front right
    33 - back right
    13 - Front Left Correct
*/

constexpr uint8_t SYNC_BYTE = 0xFF;

// BLE HID tilt-to-WASD thresholds (degrees)
constexpr float TILT_DEAD_ZONE    = 15.0f;  // angle to activate key press
constexpr float TILT_RELEASE_ZONE = 10.0f;  // angle to release key (hysteresis)

#endif
