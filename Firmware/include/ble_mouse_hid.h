#ifndef BLE_MOUSE_HID_H
#define BLE_MOUSE_HID_H

void bleMouseInit();

bool bleMouseIsConnected();

void bleMouseUpdateFromEye(float dx, float dy);