#ifndef IMU_INTERFACE_H
#define IMU_INTERFACE_H

#include <stdint.h>
#include <array>
#include "constants.h"
#include <Arduino.h>
#include "ICM_20948.h"

class IMU {
public:
    IMU(int csPin, int clkPin, int MisoPin, int MosiPin) : m_csPin(csPin), m_clkPin(clkPin), m_MisoPin(MisoPin), m_MosiPin(MosiPin) {}
    
    void begin(uint32_t spiFreq);
    void update();
    void setOutput(std::array<float, 9>& imuVals);

    float getAccX();
    float getAccY();
    float getAccZ();

    float getGyrX();
    float getGyrY();
    float getGyrZ();

    float getMagX();
    float getMagY();
    float getMagZ();
    
private:
    int m_csPin;
    int m_clkPin;
    int m_MisoPin;
    int m_MosiPin;
    ICM_20948_SPI m_icm;
    
};

#endif 
