#include <Air_Quality_Detector_ML_inferencing.h>
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
const unsigned long WARMUP_TIME_MS = 120000;

float mlInputBuffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };
size_t mlInputIndex = 0;
bool mlBufferReady = false;
bool warmupComplete = false;
unsigned long deviceStartMs = 0;
unsigned long lastWarmupPrintMs = 0;

int getMlSignalData(size_t offset, size_t length, float *outPtr) {
  memcpy(outPtr, mlInputBuffer + offset, length * sizeof(float));
  return 0;
}

void printMlResult(ei_impulse_result_t &result) {
  size_t bestIndex = 0;
  float bestValue = result.classification[0].value;

  for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > bestValue) {
      bestValue = result.classification[ix].value;
      bestIndex = ix;
    }
  }

  Serial.print(F("ML prediction: "));
  Serial.print(result.classification[bestIndex].label);
  Serial.print(F(" ("));
  Serial.print(bestValue, 5);
  Serial.println(F(")"));

  Serial.print(F("ML scores: "));
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    Serial.print(result.classification[ix].label);
    Serial.print(F("="));
    Serial.print(result.classification[ix].value, 5);
    if (ix + 1 < EI_CLASSIFIER_LABEL_COUNT) {
      Serial.print(F(", "));
    }
  }
  Serial.println();
}

void printMlWindowSummary() {
  float sgp40Min = mlInputBuffer[0];
  float sgp40Max = mlInputBuffer[0];
  float sgp40Sum = 0.0f;
  float vocMin = mlInputBuffer[1];
  float vocMax = mlInputBuffer[1];
  float vocSum = 0.0f;
  float micsMin = mlInputBuffer[2];
  float micsMax = mlInputBuffer[2];
  float micsSum = 0.0f;

  for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
    float sgp40Raw = mlInputBuffer[ix];
    float vocIndex = mlInputBuffer[ix + 1];
    float micsRaw = mlInputBuffer[ix + 2];

    sgp40Min = min(sgp40Min, sgp40Raw);
    sgp40Max = max(sgp40Max, sgp40Raw);
    sgp40Sum += sgp40Raw;

    vocMin = min(vocMin, vocIndex);
    vocMax = max(vocMax, vocIndex);
    vocSum += vocIndex;

    micsMin = min(micsMin, micsRaw);
    micsMax = max(micsMax, micsRaw);
    micsSum += micsRaw;
  }

  Serial.print(F("ML window summary: sgp40Raw avg/min/max="));
  Serial.print(sgp40Sum / EI_CLASSIFIER_RAW_SAMPLE_COUNT, 1);
  Serial.print(F("/"));
  Serial.print(sgp40Min, 0);
  Serial.print(F("/"));
  Serial.print(sgp40Max, 0);

  Serial.print(F(", vocIndex avg/min/max="));
  Serial.print(vocSum / EI_CLASSIFIER_RAW_SAMPLE_COUNT, 1);
  Serial.print(F("/"));
  Serial.print(vocMin, 0);
  Serial.print(F("/"));
  Serial.print(vocMax, 0);

  Serial.print(F(", micsRaw avg/min/max="));
  Serial.print(micsSum / EI_CLASSIFIER_RAW_SAMPLE_COUNT, 1);
  Serial.print(F("/"));
  Serial.print(micsMin, 0);
  Serial.print(F("/"));
  Serial.println(micsMax, 0);
}

void runMlInferenceIfReady() {
  if (!mlBufferReady) {
    Serial.print(F("ML collecting stable samples: "));
    Serial.print(mlInputIndex / EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
    Serial.print(F("/"));
    Serial.println(EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    return;
  }

  printMlWindowSummary();

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &getMlSignalData;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

  if (err != EI_IMPULSE_OK) {
    Serial.print(F("ML error: "));
    Serial.println((int)err);
    return;
  }

  printMlResult(result);
}

void addMlSample(uint16_t sgp40Raw, int32_t vocIndex, int micsRaw) {
  if (mlInputIndex >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    memmove(
      mlInputBuffer,
      mlInputBuffer + EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME,
      (EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) * sizeof(float)
    );
    mlInputIndex = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
  }

  size_t sampleNumber = (mlInputIndex / EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) + 1;

  mlInputBuffer[mlInputIndex++] = (float)sgp40Raw;
  mlInputBuffer[mlInputIndex++] = (float)vocIndex;
  mlInputBuffer[mlInputIndex++] = (float)micsRaw;

  Serial.print(F("ML sample "));
  Serial.print(sampleNumber);
  Serial.print(F("/"));
  Serial.print(EI_CLASSIFIER_RAW_SAMPLE_COUNT);
  Serial.print(F(": sgp40Raw="));
  Serial.print(sgp40Raw);
  Serial.print(F(", vocIndex="));
  Serial.print(vocIndex);
  Serial.print(F(", micsRaw="));
  Serial.println(micsRaw);

  if (mlInputIndex >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    mlBufferReady = true;
  }
}

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
  deviceStartMs = millis();

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

  Serial.print(F("Warming sensors for "));
  Serial.print(WARMUP_TIME_MS / 1000);
  Serial.println(F(" seconds before ML starts."));
}

void loop() {
  readSerialCommands();

  unsigned long now = millis();

  if (captureActive && captureDurationMs > 0 && now - captureStartMs >= captureDurationMs) {
    stopCapture();
  }

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = now;

  unsigned long timestamp = captureActive ? lastSampleMs - captureStartMs : lastSampleMs - deviceStartMs;

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

  if (!warmupComplete) {
    unsigned long warmupElapsed = now - deviceStartMs;

    if (warmupElapsed >= WARMUP_TIME_MS) {
      warmupComplete = true;
      Serial.println(F("Warm-up complete. ML sampling has started."));
    } else if (now - lastWarmupPrintMs >= 10000) {
      lastWarmupPrintMs = now;
      Serial.print(F("Warming up, seconds remaining: "));
      Serial.println((WARMUP_TIME_MS - warmupElapsed + 999) / 1000);
    }
  } else {
    addMlSample(sgp40Raw, vocIndex, micsRaw);
    runMlInferenceIfReady();
  }

  if (captureActive) {
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
  }
}
