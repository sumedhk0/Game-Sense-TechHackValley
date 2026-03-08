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

void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(BAUD_RATE);
    Serial.setTimeout(50);
    // while (Serial.available()) {
    //     Serial.read();
    // }
    motorsInit();
    // imu.begin(7000000); // Initialize IMU with 7MHz SPI frequency
    // bleKeyboardInit();
    Serial.println("setup done");
}

uint8_t TESTING_PIN=33;
//front center: 27
//back right: 25
//front left: 13
//right : 21
// front right: 22
// back: 32
void loop() {
    analogWrite(TESTING_PIN, 254);
    Serial.println("running");
}
