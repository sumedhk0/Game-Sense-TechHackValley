#ifndef IMU_INTERFACE_H
#define IMU_INTERFACE_H

#include <stdint.h>
#include <array>
#include "constants.h"
#include <Arduino.h>
#include "ICM_20948.h"
#include <math.h>

struct KalmanState {
    float angle = 0.0f;
    float bias  = 0.0f;
    float P[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    float Q_angle   = 0.001f;
    float Q_bias    = 0.003f;
    float R_measure = 0.03f;
};

class IMU {
public:
    IMU(int csPin, int clkPin, int MisoPin, int MosiPin) : m_csPin(csPin), m_clkPin(clkPin), m_MisoPin(MisoPin), m_MosiPin(MosiPin) {}

    void begin(uint32_t spiFreq);
    void update();
    void setOutput(std::array<float, 9>& imuVals);
    void calibrate(uint16_t numSamples = 100);
    float getPitch();
    float getRoll();

    float getAccX();
    float getAccY();
    float getAccZ();

    float getGyrX();
    float getGyrY();
    float getGyrZ();

    float getMagX();
    float getMagY();
    float getMagZ();

    void enableKalman(float qAngle = 0.001f, float qBias = 0.003f, float rMeasure = 0.03f);
    void updateKalman();
    float getKalmanPitch();
    float getKalmanRoll();

private:
    int m_csPin;
    int m_clkPin;
    int m_MisoPin;
    int m_MosiPin;
    ICM_20948_SPI m_icm;
    float m_pitchOffset = 0.0f;
    float m_rollOffset = 0.0f;

    bool m_kalmanEnabled = false;
    KalmanState m_kalmanPitch;
    KalmanState m_kalmanRoll;
    unsigned long m_lastKalmanMicros = 0;

    float kalmanUpdate(KalmanState& state, float newAngle, float newRate, float dt);
};

#endif 
