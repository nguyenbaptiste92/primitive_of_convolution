__all__ = ['register_keras_custom_object','register_alias']

import tensorflow as tf
import inspect

def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls
    
def register_alias(name: str):
    """A decorator to register a custom keras object under a given alias.
    !!! example
        ```python
        @utils.register_alias("degeneration")
        class Degeneration(tf.keras.metrics.Metric):
            pass
        ```
    """

    def register_func(cls):
        tf.keras.utils.get_custom_objects()[name] = cls
        return cls

    return register_func