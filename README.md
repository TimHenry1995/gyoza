# Gyoza 
## A package for building normalizing flow models with artificial neural networks

## Download
As this package is distributed via [PyPI](https://pypi.org/project/gyoza/) it can be installed using:
```
pip install gyoza
```

## Documentation
Detailed documentation can be found on the companion website of [read-the-docs](https://gyoza.readthedocs.io/en/latest/modules.html)

## Development
Developers can access the package via [GitHub](https://github.com/TimHenry1995/gyoza)

## Tutorial
Tutorials can be found in the [tutorials](https://github.com/TimHenry1995/gyoza/tree/main/tutorials) folder of this repository.

### Creating Models
Flow models can be created in the same way as [tensorflow models](https://keras.io/api/models/model/). Consider the following example
```
def create_model(channel_count: int = 5) -> msl.FlowLayer:

    compute_coupling_parameters = tf.keras.layers.Dense(units=channel_count)
    mask = gmm.HeaviSide(axes=[1], shape=[channel_count])
    
    network = gmf.SequentialFlowNetwork(sequence=[
        gmf.AdditiveCouplingLayer(axes=[1], 
                                    shape=[channel_count], 
                                    compute_coupling_parameters=compute_coupling_parameters, 
                                    mask=mask), 
        gmf.Shuffle(axes=[1], 
                    shape=[channel_count])
        ])

    return network

channel_count = 5
batch_size = 4

network = create_model(channel_count=channel_count)
```

### Using a Model 
A flow model can be used as follows:
```
# Forward
x = tf.reshape(tf.range(batch_size*channel_count, dtype=tf.float32), [batch_size, channel_count])
y = network(x)
```

### Saving and Loading
Models are saved and loaded in HDF5 format using the [save_weights and load_weights](https://keras.io/api/saving/weights_saving_and_loading/#saveweights-method) functions of tensorflow. The following steps shall be executed:

```
# Save existing model
path = "<your_model_path>.h5"
model.save_weights(path)

# Initialize a new instance of same architecture
new_model = create_model(channel_count=channel_count)
new_model.build(input_shape=x.shape) # Ensures model weights are initialized

# Load weights
new_model.load_weights(path)
```

Serialization via the entire model, instead of the mere weights, via [save_model](https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model) and [load_model](https://www.tensorflow.org/api_docs/python/tf/keras/saving/load_model) methods is not supported by all layers of this package and thus not recommended.
