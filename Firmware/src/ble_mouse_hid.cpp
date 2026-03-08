#include "ble_mouse_hid.h"
#include "constants.h"
#include <BLE_MOUSE_HID_H>

static BleMouse bleM("GameSense Mouse", "GameSense", 100);

void bleMouseInit() {
    bleM.begin();
}

bool bleMouseIsConnected() {
    return bleM.isConnected();
}

void bleMouseUpdateFromEye(float dx, float dy) {
    if (!bleMouseIsConnected()) {
        return;
    }

    int8_t mx = constrain((int)(dx*MOUSE_SENSITIVITY),-127,127);
    int8_t my = constrain((int)(dy*MOUSE_SENSITIVITY),-127,127);

    if (mx!=0 || my!=0) {
        bleM.move(mx,my);
    }
}