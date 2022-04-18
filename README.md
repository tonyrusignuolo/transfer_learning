I will attempt to quantize only yamnet layer weights.

I will start by loading yamnet with the yamnet.h5 weights files, then using contrib.graph_editor tool to quantize the layers.
I may have to add a quantize layer to match the yamnet/classification model on tfhub