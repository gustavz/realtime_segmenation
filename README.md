# realtime_segmenation
Realtime semantic segmentation based on [Tensorflow's DeepLab Project](https://github.com/tensorflow/models/tree/master/research/deeplab) with an extreme Focus on Performance. 
<br />
<br />
## usage
- copy `config.sample.yml` as `config.yml`, change parameters only inside there
- run `demo.py` for the original tensorflow demo
- run `test.py` to generate timeline jason files that can be investigated on `chrome://tracing/`
- run `run.py` for realtime segmentation using openCV
<br />

## current performance
- nvidia jetson tx2: **4 fps**
<br />

## HELP NEEDED
Please tell if you are able to run `run.py` on your nvidia jetson tx2 and if you get correct mask results!
I face straneg results depending on which tensorflow version I use, please tell me your setup!
