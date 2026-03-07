#include "ble_keyboard_hid.h"
#include "constants.h"
#include <BleKeyboard.h>
#include <Arduino.h>

static BleKeyboard bleKbd("GameSense Controller", "GameSense", 100);

static bool wPressed = false;
static bool aPressed = false;
static bool sPressed = false;
static bool dPressed = false;

bool switchOn;

void bleKeyboardInit() {
    bleKbd.begin();
    pinMode(SWITCH_PIN, INPUT);
    switchOn = digitalRead(SWITCH_PIN);
}



void bleKeyboardUpdateFromTilt(float pitch, float roll, bool switchOn) {
    switchOn = digitalRead(SWITCH_PIN);
    if (!bleKbd.isConnected() || !switchOn) {
        return;
    }

    // Pitch axis: W (forward) / S (backward)
    if (!wPressed && pitch > TILT_DEAD_ZONE) {
        bleKbd.press('w');
        wPressed = true;
    } else if (wPressed && pitch < TILT_RELEASE_ZONE) {
        bleKbd.release('w');
        wPressed = false;
    }

    if (!sPressed && pitch < -TILT_DEAD_ZONE) {
        bleKbd.press('s');
        sPressed = true;
    } else if (sPressed && pitch > -TILT_RELEASE_ZONE) {
        bleKbd.release('s');
        sPressed = false;
    }

    // Roll axis: A (left) / D (right)
    if (!aPressed && roll < -TILT_DEAD_ZONE) {
        bleKbd.press('a');
        aPressed = true;
    } else if (aPressed && roll > -TILT_RELEASE_ZONE) {
        bleKbd.release('a');
        aPressed = false;
    }

    if (!dPressed && roll > TILT_DEAD_ZONE) {
        bleKbd.press('d');
        dPressed = true;
    } else if (dPressed && roll < TILT_RELEASE_ZONE) {
        bleKbd.release('d');
        dPressed = false;
    }
}
