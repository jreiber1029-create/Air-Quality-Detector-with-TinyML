# Air Quality Detector with TinyML
<img src="images/IMG_3365.jpeg" width="500">
A TinyML-powered wearable air quality monitoring system designed to provide real-time environmental awareness for workers in hazardous environments.

This project combines multiple environmental sensors with on-device machine learning to classify air quality without requiring an internet connection. The system is intended as a research prototype exploring how wearable Edge AI can improve occupational health and safety by providing private, low-latency hazard detection directly on embedded hardware. Due to the academic setting of the prototype, harmful substances could not be samples. Instead coffee, multi-purpose cleaner, and vinegar were used. Because these substances share very similar chemical footprints, being able to detect and differentiate between each of them displays the potential for a device like the one I have created.  

## Features

* Real-time environmental sensing
* TinyML inference running locally on an ESP32-S3
* Multi-sensor fusion for improved air quality classification
* Offline operation with no cloud connectivity
* Designed for wearable and battery-powered deployment

## Hardware

* ESP32-S3
* Sensirion SPS30 Particulate Matter Sensor
* Sensirion SGP40 VOC Sensor
* Bosch BME280 Temperature/Humidity/Pressure Sensor
* MiCS-5524 Gas Sensor

## Machine Learning

The environmental data is collected and processed using Edge Impulse. A custom machine learning model is trained using real-world datasets and deployed directly to the microcontroller for low-power, real-time inference.

## Applications

* Construction sites
* Workshops
* Manufacturing facilities
* Industrial environments
* Occupational health research

## Project Goals

This repository documents the development of an embedded Edge AI system investigating how wearable sensing devices can provide contextual environmental awareness for workers. Rather than replacing professional monitoring equipment, the project demonstrates how low-cost sensor fusion and TinyML can be used to detect patterns associated with potentially hazardous air quality conditions while maintaining user privacy through entirely on-device processing.

## Technologies

* C++
* Arduino
* ESP32
* Edge Impulse
* TinyML
* Embedded Systems
* Sensor Fusion
* Edge AI
