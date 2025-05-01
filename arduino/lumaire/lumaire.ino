/*
LUMAIRE

Arduino code for the LUMAIRE device

The FFT part is heavily based on https://github.com/kosme/arduinoFFT/blob/master/Examples/FFT_03/FFT_03.ino
*/

/*
1/5/2025
currents:
- sleep + light off: 0.37mA
- arduino on + light off: 1.47mA
- arduino on + light max (brightness 100, rgb 120,neo pixel jewel): 6.2mA

with 1500mAH battery:
- max time: 4054 hours
- min time: 241.9 hours
*/

#include "arduinoFFT.h"
#include "ArduinoLowPower.h"
#include "Adafruit_DotStar.h"
#include "Adafruit_NeoPixel.h"
#include "BlowClassifier.h"
#include "SuckClassifier.h"

// analog inputs
const int MIC = A0;
const int VBATTERY = A2;

// toggles
bool DEBUG = false;
bool TRAIN = false;
bool SLEEP = true;

// How sensitive to be to changes in voltage for waking up
const int wakeUpMargin = 400;

// sampling
const uint16_t samples = 128;           // This value MUST ALWAYS be a power of 2
const double samplingFrequency = 5000;  // Hz, must be less than 10000 due to ADC
unsigned int sampling_period_us;
unsigned long microseconds;
int repetitions = 500;  // 2300 rep. is roughly a minute worth of looping with 5000 Hz sampling and 128 samples per loop
/*
These are the input and output vectors
Input vectors receive computed results from FFT
*/
double vReal[samples];
double vImag[samples];

// NeoPixel
const int led_pin_neo = 3;
const int max_brightness = 100;
int num_pixels = 10;
int neo_brightness = 0;

// training button
const int button_pin = 2;
int button_state = 0;

// battery
const float battery_threshold = 3.5;


/* Create FFT object */
ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, samples, samplingFrequency);

Adafruit_DotStar strip = Adafruit_DotStar(1, INTERNAL_DS_DATA, INTERNAL_DS_CLK, DOTSTAR_BGR);  // build-in led to show sleep/wake state on Trinket m0
// Argument 3 = Pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
//   NEO_RGBW    Pixels are wired for RGBW bitstream (NeoPixel RGBW products)
Adafruit_NeoPixel pixels(num_pixels, led_pin_neo, NEO_GRB + NEO_KHZ800);

void setup() {
  strip.begin();
  strip.setPixelColor(0, 0xFF0000);  // red
  if (SLEEP) {
    strip.setBrightness(1);
  } else {
    strip.setBrightness(0);
  }
  strip.show();

  pixels.begin();
  // not sure if this does anything
  pixels.setBrightness(max_brightness);
  pixels.clear();
  pixels.show();

  sampling_period_us = round(1000000 * (1.0 / samplingFrequency));
  pinMode(MIC, INPUT);
  pinMode(VBATTERY, INPUT);
  pinMode(button_pin, INPUT);
  if (DEBUG || TRAIN) {
    Serial.begin(115200);
    while (!Serial)
      ;
  }
  serialLogln("Ready");
}

void loop() {
  processAudio();
  if (SLEEP) {
    goToSleep();
  }
}

void processAudio() {
  int pred = 1;
  int prevPred = 1;
  for (int r = 0; r < repetitions; r++) {
    // SAMPLING
    microseconds = micros();
    for (int i = 0; i < samples; i++) {
      vReal[i] = analogRead(MIC);
      vImag[i] = 0;
      while (micros() - microseconds < sampling_period_us)
        ;
      microseconds += sampling_period_us;
    }

    // FFT
    FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward); /* Weigh data */
    FFT.compute(FFTDirection::Forward);                       /* Compute FFT */
    FFT.complexToMagnitude();

    int mag_len = (samples >> 1);
    float magnitude[mag_len] = { 0 };
    for (int i = 0; i < mag_len; i++) {
      magnitude[i] = (float)vReal[i];
    }

    // for generating a training set
    printLabeledData(magnitude, mag_len, 1);

    // Label mapping{'blow' : 0, 'nope' : 1, 'suck' : 2}
    prevPred = pred;
    pred = predictLabel(magnitude, mag_len, 2);
    serialLog("prediction: ");
    serialLogln(pred);
    if (pred == 0 && prevPred == 0) {
      if (batteryLow()) {
        blinkNeoPixels();
        return;
      } else {
        increaseBrightnessNeoPixels();
        serialLogln("led on");
      }
    }

    if (pred == 2) {
      turnOffNeoPixels();
      serialLogln("led off");
    }
  }
}

bool batteryLow() {
  // need to do 1:2 resistor ratio because vMax > 3.3V
  // and the board I'm currently using has 3.3V pins (Trinket M0), e.g.
  // bat - 10kO - VBATTERY - 20kO - GND
  int bat = analogRead(VBATTERY);
  float vBat = 4.95 * (bat / 1023.0);
  serialLog("Bat: ");
  serialLog(vBat);
  if (vBat < battery_threshold) {
    serialLogln(" - low");
    return true;
  }
  serialLogln(" - high");
  return false;
}

void goToSleep() {
  serialLogln("going to sleep");
  strip.setBrightness(0);
  strip.show();
  // Read the voltage at the ADC pin
  int value = analogRead(MIC);

  // Define a window around that value
  uint16_t lo = max(value - wakeUpMargin, 0);
  uint16_t hi = min(value + wakeUpMargin, UINT16_MAX);
  LowPower.attachAdcInterrupt(MIC, wakeUp, ADC_INT_OUTSIDE, lo, hi);
  LowPower.sleep();
  LowPower.detachAdcInterrupt();
  strip.setBrightness(1);
  strip.show();
}

struct labels {
  float center;
  float avg;
};

int predictLabel(float *magnitude, int mag_len, int mode) {
  if (mode == 0) {
    return blowClassifier.predict(magnitude);
  } else if (mode == 1) {
    struct labels l = getCalcLabels(magnitude, mag_len);
    float values[2] = { l.avg, l.center };
    return blowClassifier.predict(values);
  } else if (mode == 2) {
    int predS = suckClassifier.predict(magnitude);
    // suckClassifier says suck
    if (predS == 2) {
      return 2;
    }
    struct labels l = getCalcLabels(magnitude, mag_len);
    float values[2] = { l.avg, l.center };
    int predB = blowClassifier.predict(values);
    // blowClassifier says blow
    if (predB == 0) {
      return 0;
    }
    return 1;
  }
  return 0;
}

struct labels getCalcLabels(float *magnitude, int mag_len) {
  struct labels l;
  float c_nom = 0;
  float c_den = 0;
  for (int i = 0; i < mag_len; i++) {
    if (i > 2) {
      c_nom = c_nom + i * magnitude[i];
      c_den = c_den + magnitude[i];
    }
  }
  l.center = c_nom / c_den;
  l.avg = c_den / mag_len;
  return l;
}

void printLabeledData(float *magnitude, int mag_len, int mode) {
  if (TRAIN) {
    if (mode == 0) {
      for (int i = 0; i < mag_len; i++) {
        Serial.print(magnitude[i], 4);
        if (i < mag_len - 1) {
          Serial.print(",");
        }
      }
    } else if (mode == 1) {
      struct labels l = getCalcLabels(magnitude, mag_len);
      Serial.print(l.avg);
      Serial.print(",");
      Serial.print(l.center);
    }

    // for labeling training data
    button_state = digitalRead(button_pin);
    if (button_state == HIGH) {
      Serial.println(",\"suck\"");  // change to blow resp. suck
      //Serial.println(",\"blow\"");  // change to blow resp. suck
    } else {
      Serial.println(",\"nope\"");
    }
  }
}

void increaseBrightnessNeoPixels() {
  int max = 120;
  int diff = 10;
  // uint32_t color = pixels.Color(255, 100, 0, 100);
  // for (int i = 0; i < num_pixels; i++) {
  //   pixels.setPixelColor(i, color);
  // }
  // pixels.fill(color);
  for (int i = 0; i < diff; i++) {
    neo_brightness += 1;
    neo_brightness = min(neo_brightness, max);
    uint32_t color = pixels.Color(neo_brightness, neo_brightness, neo_brightness, neo_brightness);
    serialLog("Brightness :");
    serialLogln(neo_brightness);
    pixels.fill(color);
    // pixels.setBrightness(neo_brightness);
    pixels.show();
    delay(10);
  }
}

void turnOffNeoPixels() {
  neo_brightness = 0;
  pixels.clear();
  pixels.show();
}

void blinkNeoPixels() {
  neo_brightness = 100;
  uint32_t color = pixels.Color(neo_brightness, neo_brightness, neo_brightness, neo_brightness);
  serialLog("Blink brightness :");
  serialLogln(neo_brightness);
  for (int i = 0; i < 3; i++) {
    pixels.fill(color);
    pixels.show();
    delay(500);
    pixels.clear();
    pixels.show();
    delay(500);
  }
}

void serialLogln(const char *input) {
  if (DEBUG) {
    Serial.println(input);
  }
}

void serialLogln(const double input) {
  if (DEBUG) {
    Serial.println(input);
  }
}

void serialLog(const char *input) {
  if (DEBUG) {
    Serial.print(input);
  }
}

void serialLog(const double input) {
  if (DEBUG) {
    Serial.print(input);
  }
}

void wakeUp() {
}
