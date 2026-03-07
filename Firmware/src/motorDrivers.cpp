#include "motorDrivers.h"
#include "constants.h"
#include <Arduino.h>

void motorsInit() {
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        pinMode(MOTOR_PINS[i], OUTPUT);
    }
    motorsAllOff();
}   

void motorsSetAll(const std::array<uint8_t, NUM_MOTORS>& intensities) {
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        analogWrite(MOTOR_PINS[i], intensities[i]);
    }
}

void motorsAllOff() {
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        analogWrite(MOTOR_PINS[i], 0);
    }
}
