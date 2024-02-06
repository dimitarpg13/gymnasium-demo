# Keras Library Internals

Excerpt from `keras/src/layers/core/dense.py`:
```python
@keras_export("keras.layers.Dense")
class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors). The output in this case will have
    shape `(batch_size, d0, units)`.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `Dense` layer by calling `layer.enable_lora(rank)`.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel

    def call(self, inputs):
        x = ops.matmul(inputs, self.kernel)
        if self.bias is not None:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.locked = False
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(self.kernel.shape[0], rank),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel.shape[1]),
            initializer=initializers.get(b_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.kernel.trainable = False
        self._tracker.locked = True
        self.lora_enabled = True

    def save_own_variables(self, store):
        if not self.lora_enabled:
            return super().save_own_variables(store)

        kernel_value = ops.convert_to_numpy(self.kernel)
        store["0"] = kernel_value
        if self.use_bias:
            store["1"] = self.bias.numpy()

    def load_own_variables(self, store):
        if not self.lora_enabled:
            return super().load_own_variables(store)
        self._kernel.assign(store["0"])
        if self.use_bias:
            self.bias.assign(store["1"])
        self.lora_kernel_a.assign(np.zeros(self.lora_kernel_a.shape))
        self.lora_kernel_b.assign(np.zeros(self.lora_kernel_b.shape))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return {**base_config, **config}


```


Excerpt from `keras/src/initializers/random_initializers.py`:
```python
@keras_export(["keras.initializers.HeNormal", "keras.initializers.he_normal"])
class HeNormal(VarianceScaling):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = HeNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(["keras.initializers.HeUniform", "keras.initializers.he_uniform"])
class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = HeUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(
    [
        "keras.initializers.VarianceScaling",
        "keras.initializers.variance_scaling",
    ]
)
class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors.

    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:

    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use.
            One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(
        self,
        scale=1.0,
        mode="fan_in",
        distribution="truncated_normal",
        seed=None,
    ):
        if scale <= 0.0:
            raise ValueError(
                "Argument `scale` must be positive float. "
                f"Received: scale={scale}"
            )
        allowed_modes = {"fan_in", "fan_out", "fan_avg"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid `mode` argument: {mode}. "
                f"Please use one of {allowed_modes}"
            )
        distribution = distribution.lower()
        if distribution == "normal":
            distribution = "truncated_normal"
        allowed_distributions = {
            "uniform",
            "truncated_normal",
            "untruncated_normal",
        }
        if distribution not in allowed_distributions:
            raise ValueError(
                f"Invalid `distribution` argument: {distribution}."
                f"Please use one of {allowed_distributions}"
            )
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self._init_seed = seed
        self.seed = seed or random.make_default_seed()

    def __call__(self, shape, dtype=None):
        scale = self.scale
        fan_in, fan_out = compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == "truncated_normal":
            stddev = math.sqrt(scale) / 0.87962566103423978
            return random.truncated_normal(
                shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
            )
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return random.normal(
                shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
            )
        else:
            limit = math.sqrt(3.0 * scale)
            return random.uniform(
                shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed
            )

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": seed_config,
        }
```

Excerpt from `keras/src/initializers/initializer.py`
```python
@keras_export(["keras.Initializer", "keras.initializers.Initializer"])
class Initializer:
    """Initializer base class: all Keras initializers inherit from this class.

    Initializers should implement a `__call__()` method with the following
    signature:

    ```python
    def __call__(self, shape, dtype=None, **kwargs):
        # returns a tensor of shape `shape` and dtype `dtype`
        # containing values drawn from a distribution of your choice.
    ```

    Optionally, you an also implement the method `get_config()` and the class
    method `from_config` in order to support serialization -- just like with
    any Keras object.

    Here's a simple example: a random normal initializer.

    ```python
    class ExampleRandomNormal(Initializer):
        def __init__(self, mean, stddev):
            self.mean = mean
            self.stddev = stddev

        def __call__(self, shape, dtype=None, **kwargs):
            return keras.random.normal(
                shape, mean=self.mean, stddev=self.stddev, dtype=dtype
            )

        def get_config(self):  # To support serialization
            return {"mean": self.mean, "stddev": self.stddev}
    ```

    Note that we don't have to implement `from_config()` in the example above
    since the constructor arguments of the class the keys in the config returned
    by `get_config()` are the same. In this case, the default `from_config()`
    works fine.
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.
        """
        raise NotImplementedError(
            "Initializer subclasses must implement the `__call__()` method."
        )

    def get_config(self):
        """Returns the initializer's configuration as a JSON-serializable dict.

        Returns:
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Example:

        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```

        Args:
            config: A Python dictionary, the output of `get_config()`.

        Returns:
            An `Initializer` instance.
        """
        return cls(**config)

```
