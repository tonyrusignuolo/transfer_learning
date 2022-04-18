# Goal
This is an attempt to modify the source code of Yamnet to make the input shape static and quantize the model.

## Input shape
Static input shape was successfully achieved.

## Quantization
Full integer post-training quantization was attempted, but resulted in a full model quantization as opposed to only layer weight quantization. Therefore compiling for the edge tpu failed.
Next attempt is to quantize only layer weights. A possible direction would be to try the tf.contrib.graph_editor tool.
If that is unsuccessful, quantization-aware-training will be attempted. This seems like the most fruitful, but will require full retraining. My intuition is that will take a long time, but perhaps I'm wrong and that would likely be very fruitful for me.