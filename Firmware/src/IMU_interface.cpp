#include "IMU_interface.h"
#include <Arduino.h>
#include <SPI.h>

void IMU::begin(uint32_t spiFreq){
    // Initialize SPI with the pins defined in the constructor
    SPI.begin(m_clkPin, m_MisoPin, m_MosiPin, m_csPin);
    
    // Initialize the ICM-20948 library, passing the desired SPI frequency.
    // The library will cap this at 7MHz if a higher value is provided.
    m_icm.begin(m_csPin, SPI, spiFreq);
}

void IMU::update() {
    if (m_icm.dataReady()) {
        m_icm.getAGMT(); // Reads all data (Accel, Gyro, Mag, Temp)
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