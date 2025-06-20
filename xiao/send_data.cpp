#include <ArduinoBLE.h>
#include "LSM6DS3.h"

BLEService gyroService("50f74134-78ce-492b-aaa2-631fd96fa476"); // Bluetooth® Low Energy Gyro Service
BLEFloatCharacteristic gyro_X("09451b74-8500-4b3d-9090-bdf3187a98dd", BLERead | BLENotify);
BLEFloatCharacteristic gyro_Y("cc55d02b-0890-43ff-9c6b-c078d26a7d3f", BLERead | BLENotify);
BLEFloatCharacteristic gyro_Z("6161c416-e269-4d21-bbf2-afc954629dc1", BLERead | BLENotify);
//Zukünftig besser in einem, weil wohl schneller: https://docs.arduino.cc/libraries/arduinoble

LSM6DS3 myIMU(I2C_MODE, 0x6A);

float X, Y, Z;
long lastUpdate;

void setup() {
   Serial.begin(9600);
  while (!Serial);
   if (myIMU.begin() != 0) {
        Serial.println("Device error");
    } else {
        Serial.println("Device OK!");
    }
  // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");

    while (1);
  }

  // set advertised local name and service UUID:
  BLE.setLocalName("MODI_SW_Gyro");
  gyroService.addCharacteristic(gyro_X);
  gyroService.addCharacteristic(gyro_Y);
  gyroService.addCharacteristic(gyro_Z);
  BLE.setAdvertisedService(gyroService);



  // add service
  BLE.addService(gyroService);

  // start advertising
  BLE.advertise();

  Serial.println("BLE MODI SW Gyro");

}

void loop() {
   // listen for Bluetooth® Low Energy peripherals to connect:
  BLEDevice central = BLE.central();

  // if a central is connected to peripheral:
  if (central) {
    Serial.print("Connected to central: ");
    // print the central's MAC address:
    Serial.println(central.address());
  }

  while (central.connected()) {
    if (millis() > lastUpdate + 5000){
    lastUpdate = millis();
    readSensors();
    gyro_X.writeValue(X);
    gyro_Y.writeValue(Y);
    gyro_Z.writeValue(Z);
    Serial.println(String("Gyro: ") + X + ", " + Y + ", " + Z);
    }
  }
}

void readSensors(){
  X = myIMU.readFloatGyroX();
  Y = myIMU.readFloatGyroY();
  Z = myIMU.readFloatGyroZ();
}