# Lumaire

> Breathe. Glow. Lumaire

Lumaire is a light that can be controlled by breathe. Blow to turn it on, suck to turn it off.

TODO: add pictures

## Parts

- 3D printed half-sphere shaped outer part, see `3d-model/model.stl`. Needs to be printed with transparent material.
- [Adafruit Trinket M0](https://learn.adafruit.com/adafruit-trinket-m0-circuitpython-arduino/overview)
- [Adafruit Max 4466](https://www.adafruit.com/product/1063)
- [Adafruit NeoPixel Jewel](https://www.adafruit.com/product/2858)
- 1500mAh LiPo battery, e.g. [this one](https://www.bastelgarage.ch/lipo-battery-1500mah-jst-2-0-lithium-ion-polymer)
- TP4056 Lithium LiPo Battery Charging Module, either [USB C](https://www.bastelgarage.ch/tp4056-lithium-lipo-battery-charging-module-usb-c-5v-1a?search=TP4056) or [Micro USB](https://www.bastelgarage.ch/tp4056-lithium-lipo-battery-charging-module-micro-usb-5v-1a?search=TP4056)

- if you want to be fancy and also reduce the sensitivity of the lamp to the noise it makes when sliding it over a hard surface you can attach a thin layer of black filt to the bottom. You'll need to cut out the precise shape.

## Assembly

TODO: add more details

- print the 3d model
- solder the electronic parts
- load code onto the Trinket M0
- assemble everything

## Code

The code can be found in the `arduino/` directory. Basically, the audio signal from the microphone is transformed using FFT and then a machine learning model predicts whether the sound it hears is a 'blow', a 'suck' or a 'nope'. The two ML random forest classifiers that are being used are the result of the `gen_model.py` script (needs cleaning up).
