import tensorflow as tf
import numpy as np

class SaveableLayer(tf.keras.layers.Layer):
    def save(self,fn):
        weights = self.get_weights()
        writer = open(fn, 'wb')
        np_dtype = np.dtype('float32').newbyteorder('>') # Network order
        for w in weights:
            writer.write(np.array(w, dtype=np_dtype).tobytes())
        writer.close()

    def load(self,fn):
        weights = self.get_weights()
        new_weights = []
        reader = open(fn, 'rb')
        np_dtype = np.dtype('float32').newbyteorder('>') # Network order
        for w in weights:
            buffer = reader.read(w.size * 4)
            new_weights.append(np.frombuffer(buffer, dtype=np_dtype).reshape(w.shape))
        reader.close()
        self.set_weights(new_weights)

class ShapeletEncoder(SaveableLayer):
    def __init__(self,cells,classes,conv_levels=[32,32,16],return_midpoints=False,**kwargs):
        super(ShapeletEncoder, self).__init__(**kwargs)
        self.cells = cells
        self.classes = classes
        self.return_midpoints = return_midpoints
        self.conv_levels = conv_levels

    def build(self, input_shape):
        # Self constants
        self.cell_length = input_shape[1] // self.cells
        self.channels = input_shape[2]
      
        # Convolutions (shapelet classifier)
        last_level = self.cell_length
        self.classify_conv_weights = []
        self.classify_conv_bias = []
        for i in range(len(self.conv_levels)):
            level = self.conv_levels[i]
            self.classify_conv_weights.append(
                self.add_weight(name=f'conv_classify_weights_{i}', shape=(self.channels, last_level if i == 0 else 1, 1 if i == 0 else last_level, level), dtype=self.dtype, trainable=True, initializer='glorot_uniform')
            )
            self.classify_conv_bias.append(
                self.add_weight(name=f'conv_classify_bias_{i}', shape=(self.channels, level), dtype=self.dtype, trainable=True, initializer='glorot_uniform')
            )
            last_level = level
        
        self.classify_weights = self.add_weight(name='conv_classify_weights_final', shape=(self.channels, 1, last_level, self.classes), dtype=self.dtype, trainable=True, initializer='glorot_uniform')
        self.classify_bias = self.add_weight(name='conv_classify_bias_final', shape=(self.channels, self.classes), dtype=self.dtype, trainable=True, initializer='zeros')

        self.built = True

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        class_cells = []
        cell_midpoints = []
        # Store all of the midpoints for the cells(if requested) and drop the shapelets to axis
        for i in range(self.cells):
            cell_slice = inputs[:,self.cell_length * i:self.cell_length * (i+1)]
            cell_floor = tf.reduce_min(cell_slice, axis=1) 
            cell_mid = cell_floor + ((tf.reduce_max(cell_slice, axis=1) - cell_floor) / 2)
            class_cells.append(cell_slice - tf.expand_dims(cell_mid, axis=1))
            if self.return_midpoints:
                cell_midpoints.append(cell_mid)

        class_cells = tf.concat(class_cells, 1)
        if self.return_midpoints:
            cell_midpoints = tf.stack(cell_midpoints,axis=1)

        classifications = []
        # Produce classifications per channel
        for c in range(self.channels):
            last_input = tf.expand_dims(class_cells[:,:,c], axis=2)
            for i in range(len(self.conv_levels)):
                last_input = tf.nn.relu(tf.nn.conv1d(last_input, self.classify_conv_weights[i][c], self.cell_length if i == 0 else 1, 'VALID') + self.classify_conv_bias[i][c])
                # This is effectively self.cells count of dense layers laid side to side
            
            classifications.append(
                tf.nn.softmax( 
                    tf.nn.conv1d(last_input, self.classify_weights[c], 1, 'VALID') + self.classify_bias[c] 
                )
            )

        classifications = tf.concat(classifications,axis=2)
        
        if self.return_midpoints:
            return classifications, cell_midpoints
        
        return classifications

class ShapeletDecoder(SaveableLayer):
    def __init__(self,cell_length,classes,**kwargs):
        super(ShapeletDecoder, self).__init__(**kwargs)
        self.cell_length = cell_length
        self.classes = classes
 
    def build(self, input_shape):
        # Self constants
        self.channels = input_shape[2] // self.classes

        # Convolutions (shapelet decoder)
        self.decode_kernel = self.add_weight(
            name=f'conv_decode_weights', 
            shape=(self.channels, 1, self.classes, self.cell_length), 
            dtype=self.dtype, 
            trainable=True, 
            initializer='glorot_uniform')

    def call(self, inputs, midpoints=None):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        if midpoints != None and midpoints.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            midpoints = tf.cast(midpoints, dtype=self._compute_dtype_object)

        decoded_channels = []
        # Produce classifications per channel
        for c in range(self.channels):
            conv_input = inputs[:,:,c * self.classes:(c+1) * self.classes]
            conv_output = tf.nn.conv1d(conv_input, self.decode_kernel[c],1,'VALID')
            if midpoints != None:
                conv_output += midpoints[:,:,c:c+1]
            conv_output = tf.expand_dims(tf.concat(tf.unstack(conv_output, axis=1), axis=1), axis=2)
            decoded_channels.append(conv_output)
        
        decoded_channels = tf.concat(decoded_channels, axis=2)
        return decoded_channels
