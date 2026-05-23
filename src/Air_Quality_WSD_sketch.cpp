#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <SensirionI2cSps30.h>
#include "Adafruit_SGP40.h"
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#define SDA_PIN 8
#define SCL_PIN 9
#define MICS_PIN 2
#define SCK_PIN 
#define DC_PIN
#define SC_PIN
#define MOSI_PIN
#define BL_PIN

#define SEALEVELPRESSURE_HPA 1013.25

Adafruit_BME280 bme;
Adafruit_SGP40 sgp40;
SensirionI2cSps30 sps30;

#ifdef NO_ERROR
#undef NO_ERROR
#endif
#define NO_ERROR 0

String inputLine = "";
String currentLabel = "unlabeled";

bool captureActive = false;
unsigned long captureStartMs = 0;
unsigned long lastSampleMs = 0;
unsigned long captureDurationMs = 0;

const unsigned long SAMPLE_INTERVAL_MS = 1000;

uint32_t getAbsoluteHumidity(float temperature, float humidity) {
  float absoluteHumidity =
      216.7f *
      ((humidity / 100.0f) * 6.112f *
       exp((17.62f * temperature) / (243.12f + temperature)) /
       (273.15f + temperature));

  return (uint32_t)(1000.0f * absoluteHumidity);
}

void printHeader() {
  Serial.println(F("timestamp,label,temperature,pressure,humidity,absoluteHumidity,sgp40Raw,vocIndex,micsRaw,micsVoltage,mc1p0,mc2p5,mc4p0,mc10p0,nc0p5,nc1p0,nc2p5,nc4p0,nc10p0,typicalParticleSize"));
}

void startCapture(String label, unsigned long durationMs) {
  currentLabel = label;
  captureDurationMs = durationMs;
  captureStartMs = millis();
  lastSampleMs = captureStartMs;
  captureActive = true;

  printHeader();
}

void stopCapture() {
  captureActive = false;
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.equalsIgnoreCase("START")) {
    startCapture("unlabeled", 0);
    return;
  }

  if (cmd.startsWith("START ")) {
    String args = cmd.substring(6);
    args.trim();

    int firstSpace = args.indexOf(' ');

    if (firstSpace > 0) {
      String secondsString = args.substring(0, firstSpace);
      String label = args.substring(firstSpace + 1);
      secondsString.trim();
      label.trim();

      unsigned long seconds = secondsString.toInt();

      if (seconds > 0 && label.length() > 0) {
        startCapture(label, seconds * 1000UL);
        return;
      }
    }

    unsigned long seconds = args.toInt();

    if (seconds > 0) {
      startCapture("unlabeled", seconds * 1000UL);
      return;
    }

    startCapture(args, 0);
    return;
  }

  if (cmd.equalsIgnoreCase("STOP")) {
    stopCapture();
    return;
  }
}

void readSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (inputLine.length() > 0) {
        handleCommand(inputLine);
        inputLine = "";
      }
    } else {
      inputLine += c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);

  analogReadResolution(12);
  analogSetPinAttenuation(MICS_PIN, ADC_11db);

  if (!bme.begin(0x76) && !bme.begin(0x77)) {
    Serial.println(F("BME280 not found"));
    while (1) delay(100);
  }

  if (!sgp40.begin()) {
    Serial.println(F("SGP40 not found"));
    while (1) delay(100);
  }

  sps30.begin(Wire, SPS30_I2C_ADDR_69);
  sps30.stopMeasurement();
  delay(100);

  if (sps30.startMeasurement(SPS30_OUTPUT_FORMAT_OUTPUT_FORMAT_UINT16) != NO_ERROR) {
    Serial.println(F("SPS30 failed"));
    while (1) delay(100);
  }
}

void loop() {
  readSerialCommands();

  if (!captureActive) return;

  unsigned long now = millis();
  unsigned long elapsed = now - captureStartMs;

  if (captureDurationMs > 0 && elapsed >= captureDurationMs) {
    stopCapture();
    return;
  }

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = now;

  unsigned long timestamp = lastSampleMs - captureStartMs;

  float temperature = bme.readTemperature();
  float pressure = bme.readPressure() / 100.0F;
  float humidity = bme.readHumidity();
  uint32_t absoluteHumidity = getAbsoluteHumidity(temperature, humidity);

  uint16_t sgp40Raw = sgp40.measureRaw(temperature, humidity);
  int32_t vocIndex = sgp40.measureVocIndex(temperature, humidity);

  int micsRaw = analogRead(MICS_PIN);
  float micsVoltage = (micsRaw / 4095.0) * 3.3;

  uint16_t dataReadyFlag = 0;
  uint16_t mc1p0 = 0, mc2p5 = 0, mc4p0 = 0, mc10p0 = 0;
  uint16_t nc0p5 = 0, nc1p0 = 0, nc2p5 = 0, nc4p0 = 0, nc10p0 = 0;
  uint16_t typicalParticleSize = 0;

  sps30.readDataReadyFlag(dataReadyFlag);

  if (dataReadyFlag) {
    sps30.readMeasurementValuesUint16(
      mc1p0, mc2p5, mc4p0, mc10p0,
      nc0p5, nc1p0, nc2p5, nc4p0,
      nc10p0, typicalParticleSize
    );
  }

  Serial.print(timestamp); Serial.print(',');
  Serial.print(currentLabel); Serial.print(',');

  Serial.print(temperature, 3); Serial.print(',');
  Serial.print(pressure, 3); Serial.print(',');
  Serial.print(humidity, 3); Serial.print(',');
  Serial.print(absoluteHumidity); Serial.print(',');

  Serial.print(sgp40Raw); Serial.print(',');
  Serial.print(vocIndex); Serial.print(',');

  Serial.print(micsRaw); Serial.print(',');
  Serial.print(micsVoltage, 3); Serial.print(',');

  Serial.print(mc1p0); Serial.print(',');
  Serial.print(mc2p5); Serial.print(',');
  Serial.print(mc4p0); Serial.print(',');
  Serial.print(mc10p0); Serial.print(',');

  Serial.print(nc0p5); Serial.print(',');
  Serial.print(nc1p0); Serial.print(',');
  Serial.print(nc2p5); Serial.print(',');
  Serial.print(nc4p0); Serial.print(',');
  Serial.print(nc10p0); Serial.print(',');

  Serial.println(typicalParticleSize);
// hello jason
}
