#include <Arduino.h>
#include <array>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "constants.h"
#include "motorDrivers.h"
#include "IMU_interface.h"
#include "ble_keyboard_hid.h"

// Instantiate IMU with pins defined in constants.h
IMU imu(CS_PIN, SCK_PIN, MISO_PIN, MOSI_PIN);
unsigned long lastPrint = 0;
unsigned long lastBleUpdate = 0;
bool prevSwitchState = false;

void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(BAUD_RATE);
    Serial.setTimeout(50);
    while (Serial.available()) {
        Serial.read();
    }
    motorsInit();
    imu.begin(7000000);
    imu.calibrate();
    bleKeyboardInit();
    prevSwitchState = digitalRead(SWITCH_PIN);
}

void loop() {
    imu.update();
    std::array<float, 9> imuVals;
    imu.setOutput(imuVals);
    if (millis() - lastPrint > 100) {
        lastPrint = millis();
        for (float val : imuVals) {
            Serial.print(val);
            Serial.print('|');
        }
        Serial.println();
    }

    // Recalibrate on switch OFF→ON transition
    bool currentSwitch = digitalRead(SWITCH_PIN);
    if (currentSwitch && !prevSwitchState) {
        imu.calibrate();
    }
    prevSwitchState = currentSwitch;

    // Tilt-to-BLE keyboard update (~50Hz)
    if (millis() - lastBleUpdate > 20) {
        lastBleUpdate = millis();
        float pitch = imu.getPitch();
        float roll  = imu.getRoll();
        bleKeyboardUpdateFromTilt(pitch, roll);
    }

    // Flush stale data if buffer is backing up
    
    while (Serial.available() > 32) {
        Serial.read();
    }

    // Need sync byte + 8 data bytes
    if (Serial.available() < NUM_MOTORS + 1) {
        return;
    }

    // Scan for sync byte
    if (Serial.read() != SYNC_BYTE) {
        return;
    }

    // Read motor intensities
    std::array<uint8_t, NUM_MOTORS> intensities;
    if (Serial.readBytes(intensities.data(), NUM_MOTORS) == NUM_MOTORS) {
        motorsSetAll(intensities);
    }
    
}
