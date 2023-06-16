# Gyoza 
## A package for building invertible neural networks

## Download
https://pypi.org/project/gyoza/

## Documentation
https://gyoza.readthedocs.io/en/latest/modules.html

## Development
https://github.com/TimHenry1995/gyoza

## Tutorial

### Saving and Loading
Models are saved and loaded in HDF5 format using the [save_weights and load_weights](https://keras.io/api/saving/weights_saving_and_loading/#saveweights-method) functions of tensorflow. The following steps shall be executed:

```
# Save existing model
path = "<your_model_path>.h5"
model.save_weights(path)

# Initialize a new instance of same architecture
new_model = ...
new_model.build(input_shape=...) # Ensures model weights are initialized

# Load weights
new_model.load_weights(path)
```

Serialization via the entire model, instead of the mere weights, via [save_model](https://www.tensorflow.org/api_docs/python/tf/keras/saving/save_model) and [load_model](https://www.tensorflow.org/api_docs/python/tf/keras/saving/load_model) methods is not supported by all layers of this package and thus deprecated.
