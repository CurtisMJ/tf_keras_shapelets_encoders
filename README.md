# tf_keras_shapelets_encoders
A fixed interval shapelet encoder and decoder implemented in Tensorflow 2 and Keras

Dependancies are Tensorflow 2 and NumPy

This repository contains code for 2 Keras layers which comprise a deep learning based encoder and a decoder for fixed interval shapelets along a multi-channel time series graph. 
I'm not sure how useful this stuff is, I've been using it to analyze data from my weather station.

The encoder and decoder can be combined to perform unsupervised learning on both input sequences and output sequences. 
The input sequence will be divided into a number of fixed length cells, for which a certain number of "shapelet" classes will be learned.
Weights will be initialized per channel, so the total number of categories is "channels * classes"
```python
# Assuming input is shape [None,1000,None]
encoder = ShapeletEncoder(cells=100,classes=10,return_midpoints=False)
decoder = ShapeletDecoder(cell_length=10,classes=10)

encoded, midpoints = encoder(inputs)
decoded = decoder(encoded, midpoints)
```
The return_midpoints constructor parameter will cause the encoder to return a tuple of the cell classes and the midpoints of the respective shapes. The midpoints are calculated as ```min + (0.5 * range)``` of values in the cell. Only the classes are returned if the parameter is not specified or false.
The above model can use the same value for the input and target output, which will train the encoder on the input sequence's various shapes. Overfitting wisdom applies, try to get the classes as low as possible while still having decent loss.
Training a decoder on output data can also be useful for seq2seq, though the network will need to learn how to output midpoints and classes for each cell.

The topology for the encoder can be specified as a list of "levels" which are effectively linear layers applied in turn to each cell via 1-dimensional convolution layers
```python
encoder = ShapeletEncoder(cells=100,classes=10,conv_levels=[64,64,32],return_midpoints=False)
```

Both layers can be saved and loaded individually to a binary blob on disk, assuming the schema remains identical
```python
encoder.save("blob.bin")
decoder.load("blob.bin")
```
This may be useful for model building when the encoder or decoder needs to be transferred. Weights are saved in network order and can be transferred between different endian systems (e.g for inference on an ARM device)

Both layers are mixed precision aware and can perform float16 operations with float32 weights if the global policy is set accordingly.
