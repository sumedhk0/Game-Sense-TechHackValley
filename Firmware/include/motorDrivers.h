#ifndef MOTOR_DRIVERS_H
#define MOTOR_DRIVERS_H

#include <stdint.h>
#include <array>
#include "constants.h"

void motorsInit();
void motorsSetAll(const std::array<uint8_t, NUM_MOTORS>& intensities);
void motorsAllOff();

#endif
