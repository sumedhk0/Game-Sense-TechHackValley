#include "IMU_interface.h"
#include <Arduino.h>
#include <SPI.h>

ICM_20948_Status_e IMU::begin(uint32_t spiFreq){
    Serial.println("--- IMU DIAGNOSTIC BEGIN ---");
    Serial.print("SPI Pins - CS:"); Serial.print(m_csPin);
    Serial.print(" SCK:"); Serial.print(m_clkPin);
    Serial.print(" MISO:"); Serial.print(m_MisoPin);
    Serial.print(" MOSI:"); Serial.println(m_MosiPin);
    Serial.print("SPI Frequency: "); Serial.println(spiFreq);

    SPI.begin(m_clkPin, m_MisoPin, m_MosiPin, m_csPin);

    ICM_20948_Status_e result = m_icm.begin(m_csPin, SPI, spiFreq);

    Serial.print("ICM-20948 begin() status: ");
    Serial.println(m_icm.statusString(result));

    if (result == ICM_20948_Stat_Ok) {
        Serial.println("IMU initialized successfully.");
    } else {
        uint8_t whoami = m_icm.getWhoAmI();
        Serial.print("WHO_AM_I register reads: 0x");
        Serial.println(whoami, HEX);

        bool connected = m_icm.isConnected();
        Serial.print("isConnected(): ");
        Serial.println(connected ? "true" : "false");

        if (whoami == 0xEA) {
            Serial.println("DIAGNOSIS: Chip responds with correct ID (0xEA).");
            Serial.println("  Init sequence failed after ID check. Likely SOFTWARE issue.");
            Serial.println("  Try: enable imu.enableDebugging(Serial) or lower SPI freq.");
        } else if (whoami == 0x00 || whoami == 0xFF) {
            Serial.println("DIAGNOSIS: WHO_AM_I reads 0x00/0xFF - no SPI communication.");
            Serial.println("  Likely HARDWARE issue. Check:");
            Serial.println("  - Wiring: CS, SCK, MISO, MOSI connections");
            Serial.println("  - Power: Is the IMU getting 3.3V?");
            Serial.println("  - Solder joints on the breakout board");
        } else {
            Serial.print("DIAGNOSIS: Unexpected WHO_AM_I (expected 0xEA, got 0x");
            Serial.print(whoami, HEX);
            Serial.println("). Wrong chip or SPI bus noise.");
        }

        if (m_csPin == 2) {
            Serial.println("WARNING: CS is on GPIO 2 (ESP32 strapping pin).");
            Serial.println("  This can prevent boot if held LOW. Consider changing CS pin.");
        }
    }
    Serial.println("--- IMU DIAGNOSTIC END ---");
    return result;
}

void IMU::update() {
    if (m_icm.dataReady()) {
        m_icm.getAGMT();
        if (m_icm.status != ICM_20948_Stat_Ok) {
            Serial.print("IMU read error: ");
            Serial.println(m_icm.statusString());
        }
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

bool IMU::isConnected() { return m_icm.isConnected(); }
void IMU::enableDebugging(Stream &port) { m_icm.enableDebugging(port); }

void IMU::enableKalman(float qAngle, float qBias, float rMeasure) {
    m_kalmanPitch = KalmanState();
    m_kalmanRoll  = KalmanState();
    m_kalmanPitch.Q_angle = qAngle;
    m_kalmanPitch.Q_bias  = qBias;
    m_kalmanPitch.R_measure = rMeasure;
    m_kalmanRoll.Q_angle = qAngle;
    m_kalmanRoll.Q_bias  = qBias;
    m_kalmanRoll.R_measure = rMeasure;

    float ax = m_icm.accX(), ay = m_icm.accY(), az = m_icm.accZ();
    m_kalmanPitch.angle = atan2f(ay, sqrtf(ax * ax + az * az)) * 180.0f / M_PI;
    m_kalmanRoll.angle  = atan2f(-ax, az) * 180.0f / M_PI;

    m_lastKalmanMicros = micros();
    m_kalmanEnabled = true;
}

float IMU::kalmanUpdate(KalmanState& s, float newAngle, float newRate, float dt) {
    // Predict
    float rate = newRate - s.bias;
    s.angle += dt * rate;

    s.P[0][0] += dt * (dt * s.P[1][1] - s.P[0][1] - s.P[1][0] + s.Q_angle);
    s.P[0][1] -= dt * s.P[1][1];
    s.P[1][0] -= dt * s.P[1][1];
    s.P[1][1] += s.Q_bias * dt;

    // Update
    float S = s.P[0][0] + s.R_measure;
    float K0 = s.P[0][0] / S;
    float K1 = s.P[1][0] / S;

    float y = newAngle - s.angle;
    s.angle += K0 * y;
    s.bias  += K1 * y;

    float P00_temp = s.P[0][0];
    float P01_temp = s.P[0][1];
    s.P[0][0] -= K0 * P00_temp;
    s.P[0][1] -= K0 * P01_temp;
    s.P[1][0] -= K1 * P00_temp;
    s.P[1][1] -= K1 * P01_temp;

    return s.angle;
}

void IMU::updateKalman() {
    if (!m_kalmanEnabled) return;

    unsigned long now = micros();
    float dt = (now - m_lastKalmanMicros) / 1000000.0f;
    m_lastKalmanMicros = now;
    if (dt <= 0.0f || dt > 1.0f) return;

    float ax = m_icm.accX(), ay = m_icm.accY(), az = m_icm.accZ();
    float accPitch = atan2f(ay, sqrtf(ax * ax + az * az)) * 180.0f / M_PI;
    float accRoll  = atan2f(-ax, az) * 180.0f / M_PI;

    float gyrXrate = m_icm.gyrX();
    float gyrYrate = m_icm.gyrY();

    kalmanUpdate(m_kalmanPitch, accPitch, gyrXrate, dt);
    kalmanUpdate(m_kalmanRoll,  accRoll,  gyrYrate, dt);
}

float IMU::getKalmanPitch() { return m_kalmanPitch.angle; }
float IMU::getKalmanRoll()  { return m_kalmanRoll.angle;  }
