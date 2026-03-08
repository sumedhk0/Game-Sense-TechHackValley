#include "IMU_interface.h"
#include <Arduino.h>
#include <SPI.h>

void IMU::begin(uint32_t spiFreq){
    SPI.begin(m_clkPin, m_MisoPin, m_MosiPin, m_csPin);
    m_icm.begin(m_csPin, SPI, spiFreq);
}

void IMU::update() {
    if (m_icm.dataReady()) {
        m_icm.getAGMT();
    }
}

void IMU::setOutput(std::array<float, 9>& imuVals) {
    imuVals[0] = m_icm.accX();
    imuVals[1] = m_icm.accY();
    imuVals[2] = m_icm.accZ();
    imuVals[3] = m_icm.gyrX();
    imuVals[4] = m_icm.gyrY();
    imuVals[5] = m_icm.gyrZ();
    imuVals[6] = m_icm.magX();
    imuVals[7] = m_icm.magY();
    imuVals[8] = m_icm.magZ();

}


float IMU::getAccX() { return m_icm.accX(); }
float IMU::getAccY() { return m_icm.accY(); }
float IMU::getAccZ() { return m_icm.accZ(); }

float IMU::getGyrX() { return m_icm.gyrX(); }
float IMU::getGyrY() { return m_icm.gyrY(); }
float IMU::getGyrZ() { return m_icm.gyrZ(); }

float IMU::getMagX() { return m_icm.magX(); }
float IMU::getMagY() { return m_icm.magY(); }
float IMU::getMagZ() { return m_icm.magZ(); }

void IMU::calibrate(uint16_t numSamples) {
    float sumAx = 0, sumAy = 0, sumAz = 0;
    uint16_t collected = 0;

    for (uint16_t i = 0; i < numSamples; i++) {
        delay(5);
        if (m_icm.dataReady()) {
            m_icm.getAGMT();
            sumAx += m_icm.accX();
            sumAy += m_icm.accY();
            sumAz += m_icm.accZ();
            collected++;
        }
    }

    if (collected > 0) {
        float avgAx = sumAx / collected;
        float avgAy = sumAy / collected;
        float avgAz = sumAz / collected;
        m_pitchOffset = atan2(-avgAx, sqrt(avgAy * avgAy + avgAz * avgAz)) * 180.0f / M_PI;
        m_rollOffset  = atan2(avgAy, avgAz) * 180.0f / M_PI;
    }
}

float IMU::getPitch() {
    float ax = m_icm.accX(), ay = m_icm.accY(), az = m_icm.accZ();
    return atan2(-ax, sqrt(ay * ay + az * az)) * 180.0f / M_PI - m_pitchOffset;
}

float IMU::getRoll() {
    float ay = m_icm.accY(), az = m_icm.accZ();
    return atan2(ay, az) * 180.0f / M_PI - m_rollOffset;
}