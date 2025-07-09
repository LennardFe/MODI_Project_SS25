#include <ArduinoBLE.h> //https://docs.arduino.cc/libraries/arduinoble
#include "LSM6DS3.h"


#define NUMBER_OF_SENSORS 3

union multi_sensor_data
{
  struct __attribute__( ( packed ) )
  {
    float values[NUMBER_OF_SENSORS];
  };
  uint8_t bytes[ NUMBER_OF_SENSORS * sizeof( float ) ];
};

union multi_sensor_data gyroData;
union multi_sensor_data accelData;

//--------------------------------------------------------------------------------
// BLE
//--------------------------------------------------------------------------------

BLEService imuService("50f74134-78ce-492b-aaa2-631fd96fa476"); // Bluetooth® Low Energy Gyro Service
BLECharacteristic gyroCharacteristic("09451b74-8500-4b3d-9090-bdf3187a98dd", BLERead | BLENotify, sizeof gyroData.bytes);
BLECharacteristic accelCharacteristic("cc55d02b-0890-43ff-9c6b-c078d26a7d3f", BLERead | BLENotify, sizeof accelData.bytes);
LSM6DS3 myIMU(I2C_MODE, 0x6A);


void setup() {
  // Optional: Nur aktivieren, wenn du Debugging über Serial brauchst
  // Serial.begin(9600);
  // delay(2000); // Warten, bis USB bereit ist (wenn nötig)
  // while (!Serial); // Nur verwenden, wenn du wirklich auf Serial wartest

  // --------------------
  // Initialisiere IMU
  // --------------------
  bool imuReady = false;
  for (int attempts = 0; attempts < 50; attempts++) {
    if (myIMU.begin() == 0) {
      imuReady = true;
      break;
    }
    delay(500); // Warte und versuche es erneut
  }

  if (!imuReady) {
    // Wenn IMU nach mehreren Versuchen nicht startet, stoppen wir alles
    // Optional: LED blinken lassen oder anderen Hinweis geben
    while (1); // Dauerhaft anhalten
  }

  // --------------------
  // Starte BLE
  // --------------------
  if (!BLE.begin()) {
    // Optional: Debug-Meldung
    // Serial.println("BLE Start fehlgeschlagen");
    while (1); // BLE konnte nicht gestartet werden
  }

  // BLE-Konfiguration
  BLE.setLocalName("MODI_SW_IMU");

  imuService.addCharacteristic(gyroCharacteristic);
  imuService.addCharacteristic(accelCharacteristic);
  BLE.setAdvertisedService(imuService);
  BLE.addService(imuService);
  BLE.advertise();

  // Optional: Debug-Meldung
  // Serial.println("BLE gestartet, wartet auf Verbindung...");
}


void loop() {
   // listen for Bluetooth® Low Energy peripherals to connect:
  BLEDevice central = BLE.central();


  while (central.connected()) {
    readGyroSensors();
    gyroCharacteristic.writeValue(gyroData.bytes, sizeof gyroData.bytes);
    readAccelSensors();
    accelCharacteristic.writeValue(accelData.bytes, sizeof accelData.bytes);
  }
}

void readGyroSensors(){
  gyroData.values[0] = myIMU.readFloatGyroX();
  gyroData.values[1] = myIMU.readFloatGyroY();
  gyroData.values[2] = myIMU.readFloatGyroZ();
}

void readAccelSensors(){
  accelData.values[0] = myIMU.readFloatAccelX();
  accelData.values[1] = myIMU.readFloatAccelY();
  accelData.values[2] = myIMU.readFloatAccelZ();
}