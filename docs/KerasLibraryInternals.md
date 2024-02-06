# Keras Library Internals

## Contents
[The Layers in Keras](#the-layers-in-keras)

[Initializers in Keras](#initializers-in-keras)


## <a id="the-layers-in-keras"></a>The Layers in Keras 


Excerpt from `keras/src/layers/layer.py`:
```python
@keras_export(["keras.Layer", "keras.layers.Layer"])
class Layer(BackendLayer, Operation):
    """This is the class from which all layers inherit.

    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables). State can be
    created:

    * in `__init__()`, for instance via `self.add_weight()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the layer, and supplies the shape(s) of the input(s),
      which may not have been known at initialization time.

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a layer and then treat it as a callable.

    Args:
        trainable: Boolean, whether the layer's variables should be trainable.
        name: String name of the layer.
        dtype: The dtype of the layer's computations and weights. Can also be a
            `keras.mixed_precision.DTypePolicy`,
            which allows the computation and
            weight dtype to differ. Defaults to `None`. `None` means to use
            `keras.mixed_precision.dtype_policy()`,
            which is a `float32` policy unless set to different value
            (via `keras.mixed_precision.set_dtype_policy()`).

    Attributes:
        name: The name of the layer (string).
        dtype: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
        variable_dtype: Dtype of the layer's weights.
        compute_dtype: The dtype of the layer's computations.
            Layers automatically cast inputs to this dtype, which causes
            the computations and output to also be in this dtype.
            When mixed precision is used with a
            `keras.mixed_precision.DTypePolicy`, this will be different
            than `variable_dtype`.
        trainable_weights: List of variables to be included in backprop.
        non_trainable_weights: List of variables that should not be
            included in backprop.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        trainable: Whether the layer should be trained (boolean), i.e.
            whether its potentially-trainable weights should be returned
            as part of `layer.trainable_weights`.
        input_spec: Optional (list of) `InputSpec` object(s) specifying the
            constraints on inputs that can be accepted by the layer.

    We recommend that descendants of `Layer` implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
        that do not depend on input shapes, using `add_weight()`,
        or other state.
    * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`, or other
        state. `__call__()` will automatically build the layer
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the layer to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
            2. `mask` (boolean tensor encoding masked timesteps in the input,
                used e.g. in RNN layers).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` and `mask` if the layer need them.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this layer. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the layer or a model that contains this layer.

    Examples:

    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).

    ```python
    class SimpleDense(Layer):
        def __init__(self, units=32):
            super().__init__()
            self.units = units

        # Create the state of the layer (weights)
        def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel",
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        # Defines the computation
        def call(self, inputs):
            return ops.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```

    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:

    ```python
    class ComputeSum(Layer):

      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="total",
          )

      def call(self, inputs):
          self.total.assign(self.total + ops.sum(inputs))
          return self.total

    my_sum = ComputeSum(2)
    x = ops.ones((2, 2))
    y = my_sum(x)

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```
    """

    def __new__(cls, *args, **kwargs):
        # Wrap the user-provided build method in the build_decorator
        # to add name scope support and serialization support.
        obj = super().__new__(cls, *args, **kwargs)

        original_build_method = obj.build

        @wraps(original_build_method)
        def build_wrapper(*args, **kwargs):
            with backend.name_scope(obj.name, caller=obj):
                original_build_method(*args, **kwargs)
            # Record build config.
            signature = inspect.signature(original_build_method)
            obj._build_shapes_dict = signature.bind(*args, **kwargs).arguments
            # Set built, post build actions, and lock state.
            obj.built = True
            obj._post_build()
            obj._lock_state()

        obj.build = build_wrapper
        return obj

    def __init__(
        self,
        *,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs,
    ):
        BackendLayer.__init__(self)
        self._lock = False
        Operation.__init__(self, name=name)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        input_dim_arg = kwargs.pop("input_dim", None)
        if input_dim_arg is not None:
            input_shape_arg = (input_dim_arg,)
        else:
            input_shape_arg = kwargs.pop("input_shape", None)
        if input_shape_arg is not None:
            warnings.warn(
                "Do not pass an `input_shape`/`input_dim` argument to "
                "a layer. When using Sequential models, "
                "prefer using an `Input(shape)` object as the "
                "first layer in the model instead.",
                stacklevel=2,
            )
            self._input_shape_arg = input_shape_arg
        if kwargs:
            raise ValueError(
                "Unrecognized keyword arguments "
                f"passed to {self.__class__.__name__}: {kwargs}"
            )

        self.built = False
        self.dtype_policy = mixed_precision.resolve_policy(dtype)
        self.autocast = autocast
        self._input_spec = None
        self._called = False
        self.supports_jit = True

        self._trainable = trainable
        self._losses = []
        self._loss_ids = set()

        self._call_signature = inspect.signature(self.call)
        call_signature_parameters = [
            p.name for p in self._call_signature.parameters.values()
        ]
        self._call_has_training_arg = "training" in call_signature_parameters
        self._call_has_mask_arg = "mask" in call_signature_parameters

        self._supports_masking = not utils.is_default(self.compute_mask)
        # Whether to automatically convert (+ auto-cast) inputs to `call()`.
        self._convert_input_args = True
        # Whether to allow non-tensors as positional arguments in `call()`.
        self._allow_non_tensor_positional_args = False
        # Dict of shapes that were used to call `build()`.
        self._build_shapes_dict = None
        self._initializer_tracker()

    @tracking.no_automatic_dependency_tracking
    def _initializer_tracker(self):
        if hasattr(self, "_tracker"):
            return

        trainable_variables = []
        non_trainable_variables = []
        layers = []
        metrics = []
        seed_generators = []
        self._tracker = tracking.Tracker(
            {
                "trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and x.trainable,
                    trainable_variables,
                ),
                "non_trainable_variables": (
                    lambda x: isinstance(x, backend.Variable)
                    and not x.trainable,
                    non_trainable_variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), metrics),
                "layers": (
                    lambda x: isinstance(x, Layer)
                    and not isinstance(x, Metric),
                    layers,
                ),
                "seed_generators": (
                    lambda x: isinstance(x, backend.random.SeedGenerator),
                    seed_generators,
                ),
            }
        )
        if backend.backend() == "tensorflow":
            # Remove attribute tracking for lists (TF-specific attribute)
            _self_setattr_tracking = getattr(
                self, "_self_setattr_tracking", True
            )
            self._self_setattr_tracking = False

        self._trainable_variables = trainable_variables
        self._non_trainable_variables = non_trainable_variables
        self._layers = layers
        self._metrics = metrics
        self._seed_generators = seed_generators

        if backend.backend() == "tensorflow":
            # Reset attribute tracking (TF-specific)
            self._self_setattr_tracking = _self_setattr_tracking

    @property
    def input_spec(self):
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value):
        self._input_spec = value

    @utils.default
    def build(self, input_shape):
        self._check_super_called()
        if utils.is_default(self.build) and might_have_unbuilt_state(self):
            warnings.warn(
                f"`build()` was called on layer '{self.name}', however "
                "the layer does not have a `build()` method implemented "
                "and it looks like it has unbuilt state. This will cause "
                "the layer to be marked as built, despite not being "
                "actually built, which may cause failures down the line. "
                "Make sure to implement a proper `build()` method."
            )
        self.built = True

    def _lock_state(self):
        """Prevent further state updates, called automatically in `build()`."""
        if not self._tracker.locked:
            self._tracker.lock(
                msg=(
                    "You cannot add new elements of state "
                    "(variables or sub-layers) "
                    "to a layer that is already built. All state "
                    "must be created in the `__init__()` method or "
                    "in the `build()` method."
                )
            )

    def get_build_config(self):
        """Returns a dictionary with the layer's input shape.

        This method returns a config dict that can be used by
        `build_from_config(config)` to create all states (e.g. Variables and
        Lookup tables) needed by the layer.

        By default, the config only contains the input shape that the layer
        was built with. If you're writing a custom layer that creates state in
        an unusual way, you should override this method to make sure this state
        is already created when Keras attempts to load its value upon model
        loading.

        Returns:
            A dict containing the input shape associated with the layer.
        """
        if self._build_shapes_dict is not None:
            if len(self._build_shapes_dict) == 1:
                return {
                    "input_shape": tuple(self._build_shapes_dict.values())[0],
                }
            else:
                return {"shapes_dict": self._build_shapes_dict}

    def build_from_config(self, config):
        """Builds the layer's states with the supplied config dict.

        By default, this method calls the `build(config["input_shape"])` method,
        which creates weights based on the layer's input shape in the supplied
        config. If your config contains other information needed to load the
        layer's state, you should override this method.

        Args:
            config: Dict containing the input shape associated with this layer.
        """
        if config:
            if "input_shape" in config:
                self.build(config["input_shape"])
            elif "shapes_dict" in config:
                self.build(**config["shapes_dict"])
            self.built = True

    def add_variable(
        self,
        shape,
        initializer,
        dtype=None,
        trainable=True,
        regularizer=None,
        constraint=None,
        name=None,
    ):
        """Add a weight variable to the layer.

        Alias of `add_weight()`.
        """
        return self.add_weight(
            shape=shape,
            initializer=initializer,
            dtype=dtype,
            trainable=trainable,
            regularizer=regularizer,
            constraint=constraint,
            name=name,
        )

    def add_weight(
        self,
        shape=None,
        initializer=None,
        dtype=None,
        trainable=True,
        regularizer=None,
        constraint=None,
        name=None,
    ):
        """Add a weight variable to the layer.

        Args:
            shape: Shape tuple for the variable.
                Must be fully-defined (no `None` entries).
                Defaults to `()` (scalar) if unspecified.
            initializer: Initializer object to use to
                populate the initial variable value,
                or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified,
                defaults to `"glorot_uniform"`
                for floating-point variables and to `"zeros"`
                for all other types (e.g. int, bool).
            dtype: Dtype of the variable to create,
                e.g. `"float32"`. If unspecified,
                defaults to the layer's
                variable dtype (which itself defaults to
                `"float32"` if unspecified).
            trainable: Boolean, whether the variable should
                be trainable via backprop or whether its
                updates are managed manually.
            constraint: Contrainst object to call on the
                variable after any optimizer update,
                or string name of a built-in constraint.
            name: String name of the variable. Useful
                for debugging purposes.
        """
        self._check_super_called()
        if shape is None:
            shape = ()
        if dtype is not None:
            dtype = backend.standardize_dtype(dtype)
        else:
            dtype = self.variable_dtype
        if initializer is None:
            if "float" in dtype:
                initializer = "glorot_uniform"
            else:
                initializer = "zeros"
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                name=name,
            )
        # Will be added to layer.losses
        variable.regularizer = regularizers.get(regularizer)
        variable.constraint = constraints.get(constraint)
        self._track_variable(variable)
        return variable

    @property
    def trainable(self):
        """Settable boolean, whether this layer should be trainable or not."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Sets trainable attribute for the layer and its sublayers.

        When this value is changed during training (e.g. with a
        `Callback`) you need to call the parent
        `Model.make_train_function` with `force=True` in order to
        recompile the training graph.

        Args:
            value: Boolean with the desired state for the layer's trainable
                attribute.
        """
        value = bool(value)
        self._trainable = value
        for v in self._trainable_variables:
            v.trainable = value
        for layer in self._layers:
            layer.trainable = value

    @property
    def variables(self):
        """List of all layer state, including random seeds.

        This extends `layer.weights` to include all state used by the layer
        including `SeedGenerator`s.

        Note that metrics variables are not included here, use
        `metrics_variables` to visit all the metric variables.
        """
        # Return all `Variables` associate with the layer including metrics
        # and random seeds. Also deduplicate them.
        variables = []
        seen_ids = set()
        for v in self._trainable_variables + self._non_trainable_variables:
            if id(v) not in seen_ids:
                variables.append(v)
                seen_ids.add(id(v))
        for sg in self._seed_generators:
            variables.append(sg.state)
        for layer in self._layers:
            for v in layer.variables:
                if id(v) not in seen_ids:
                    variables.append(v)
                    seen_ids.add(id(v))
        return variables

    @property
    def trainable_variables(self):
        """List of all trainable layer state.

        This is equivalent to `layer.trainable_weights`.
        """
        if not self.trainable:
            return []
        return [v for v in self.variables if v.trainable]

    @property
    def non_trainable_variables(self):
        """List of all non-trainable layer state.

        This extends `layer.non_trainable_weights` to include all state used by
        the layer including state for metrics and `SeedGenerator`s.
        """
        if not self.trainable:
            return self.variables
        return [v for v in self.variables if not v.trainable]

    @property
    def weights(self):
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        # Return only `Variables` directly owned by layers and sub-layers.
        # Also deduplicate them.
        weights = []
        seen_ids = set()
        for w in self._trainable_variables + self._non_trainable_variables:
            if id(w) not in seen_ids:
                weights.append(w)
                seen_ids.add(id(w))
        for layer in self._layers:
            for w in layer.weights:
                if id(w) not in seen_ids:
                    weights.append(w)
                    seen_ids.add(id(w))
        return weights

    @property
    def trainable_weights(self):
        """List of all trainable weight variables of the layer.

        These are the weights that get updated by the optimizer during training.
        """
        if not self.trainable:
            return []
        return [v for v in self.weights if v.trainable]

    @property
    def non_trainable_weights(self):
        """List of all non-trainable weight variables of the layer.

        These are the weights that should not be updated by the optimizer during
        training. Unlike, `layer.non_trainable_variables` this excludes metric
        state and random seeds.
        """
        if not self.trainable:
            return self.weights
        return [v for v in self.weights if not v.trainable]

    @property
    def metrics_variables(self):
        """List of all metric variables."""
        vars = []
        for metric in self._metrics:
            vars.extend(metric.variables)
        for layer in self._layers:
            for metric in layer._metrics:
                vars.extend(metric.variables)
        return vars

    def get_weights(self):
        """Return the values of `layer.weights` as a list of NumPy arrays."""
        return [v.numpy() for v in self.weights]

    def set_weights(self, weights):
        """Sets the values of `layer.weights` from a list of NumPy arrays."""
        layer_weights = self.weights
        if len(layer_weights) != len(weights):
            raise ValueError(
                f"You called `set_weights(weights)` on layer '{self.name}' "
                f"with a weight list of length {len(weights)}, but the layer "
                f"was expecting {len(layer_weights)} weights."
            )
        for variable, value in zip(layer_weights, weights):
            if variable.shape != value.shape:
                raise ValueError(
                    f"Layer {self.name} weight shape {variable.shape} "
                    "is not compatible with provided weight "
                    f"shape {value.shape}."
                )
            variable.assign(value)

    @property
    def dtype(self):
        """Alias of `layer.variable_dtype`."""
        return self.variable_dtype

    @property
    def compute_dtype(self):
        """The dtype of the computations performed by the layer."""
        return self.dtype_policy.compute_dtype

    @property
    def variable_dtype(self):
        """The dtype of the state (weights) of the layer."""
        return self.dtype_policy.variable_dtype

    @property
    def input_dtype(self):
        """The dtype layer inputs should be converted to."""
        return self.dtype_policy.compute_dtype

    @property
    def supports_masking(self):
        """Whether this layer supports computing a mask using `compute_mask`."""
        return self._supports_masking

    @supports_masking.setter
    def supports_masking(self, value):
        self._supports_masking = value

    @utils.default
    def compute_mask(self, inputs, previous_mask):
        return previous_mask

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        self._check_super_called()
        self._called = True

        #####################################
        # 1. Convert any array arguments to tensors of correct dtype.
        def maybe_convert(x):
            if backend.is_tensor(x):
                if (
                    self.autocast
                    and backend.is_float_dtype(x.dtype)
                    and x.dtype != self.input_dtype
                ):
                    x = backend.cast(x, dtype=self.input_dtype)
                return x
            elif isinstance(x, backend.KerasTensor):
                if (
                    self.autocast
                    and backend.is_float_dtype(x.dtype)
                    and x.dtype != self.input_dtype
                ):
                    x.dtype = self.input_dtype
                return x
            elif hasattr(x, "__array__"):
                return backend.convert_to_tensor(x, dtype=self.input_dtype)
            return x

        # Used to avoid expensive `tree` operations in the most common case.
        if (
            kwargs
            or len(args) != 1
            or not backend.is_tensor(args[0])
            or backend.standardize_dtype(args[0].dtype) != self.input_dtype
        ) and self._convert_input_args:
            args = tree.map_structure(maybe_convert, args)
            kwargs = tree.map_structure(maybe_convert, kwargs)

        ##########################################################
        # 2. Enforce that only tensors can be passed positionally.
        if not self._allow_non_tensor_positional_args:
            for arg in tree.flatten(args):
                if not isinstance(arg, KerasTensor) and not backend.is_tensor(
                    arg
                ):
                    raise ValueError(
                        "Only input tensors may be passed as "
                        "positional arguments. The following argument value "
                        f"should be passed as a keyword argument: {arg} "
                        f"(of type {type(arg)})"
                    )

        # Caches info about `call()` signature, args, kwargs.
        call_spec = CallSpec(self._call_signature, args, kwargs)

        ############################################
        # 3. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(call_spec.first_arg)

        ################
        # 4. Call build
        with backend.name_scope(self.name, caller=self):
            self._maybe_build(call_spec)

        ##########################
        # 5. Infer training value
        # Training phase for `Layer.call` is set via (in order of priority):
        # (1) The `training` argument passed to this `Layer.call`, if not None
        # (2) The training argument of an outer `Layer.call`.
        # (4) Any non-None default value for `training` in the call signature
        # (5) False (treating the layer as if it's in inference)

        # Maintains info about the `Layer.call` stack
        # across nested calls.
        call_context = self._get_call_context()

        # This is the value explicity passed by the user
        training = call_spec.user_arguments_dict.get("training", None)
        if training is None:
            # Wasn't passed explicitly: use context value
            training = call_context.training
            if training is None:
                # Get signature default value
                training = call_spec.arguments_dict.get("training", None)
        call_context.training = training
        if self._call_has_training_arg and training is not None:
            # Only populate arg if it has a concrete value
            kwargs["training"] = training

        ##############################
        # 6. Populate mask argument(s)
        if len(call_spec.tensor_arguments_dict) == 1:
            if (
                "mask" in call_spec.argument_names
                and call_spec.arguments_dict["mask"] is None
            ):
                arg_name = list(call_spec.tensor_arguments_dict.keys())[0]
                only_tensor_arg = call_spec.tensor_arguments_dict[arg_name]
                mask = tree.map_structure(
                    lambda x: getattr(x, "_keras_mask", None),
                    only_tensor_arg,
                )
                kwargs["mask"] = mask
        elif len(call_spec.tensor_arguments_dict) > 1:
            for k, v in call_spec.tensor_arguments_dict.items():
                expected_mask_arg_name = f"{k}_mask"
                if expected_mask_arg_name in call_spec.argument_names:
                    if call_spec.arguments_dict[expected_mask_arg_name] is None:
                        mask = tree.map_structure(
                            lambda x: getattr(x, "_keras_mask", None), v
                        )
                        kwargs[expected_mask_arg_name] = mask

        ####################
        # 7. Call the layer.
        try:
            with backend.name_scope(self.name, caller=self):
                current_scope = backend.get_autocast_scope()
                new_scope = None
                if current_scope is not None:
                    # Clear or update the current scope if necessary.
                    if not self.autocast:
                        new_scope = backend.AutocastScope(None)
                    elif not backend.is_float_dtype(self.compute_dtype):
                        # Some preprocessing layers might have a non-float
                        # dtype, we should not autocast in this case.
                        new_scope = backend.AutocastScope(None)
                    elif current_scope.dtype != self.compute_dtype:
                        new_scope = backend.AutocastScope(self.compute_dtype)
                elif self.compute_dtype != self.variable_dtype:
                    # Enter a new scope if our dtypes are "mixed".
                    new_scope = backend.AutocastScope(self.compute_dtype)

                if new_scope is not None:
                    with new_scope:
                        outputs = super().__call__(*args, **kwargs)
                else:
                    outputs = super().__call__(*args, **kwargs)
                # Change the layout for the layer output if needed.
                # This is useful for relayout intermediate tensor in the model
                # to achieve the optimal performance.
                distribution = distribution_lib.distribution()
                if distribution is not None:
                    current_layer_path = current_path()
                    current_layer_path += "/output"
                    layout = distribution.get_tensor_layout(current_layer_path)
                    if layout:
                        outputs = distribution_lib.distribute_tensor(
                            outputs, layout
                        )

                if not self.built:
                    self.built = True
                # Record activity regularizer loss.
                if self.activity_regularizer is not None:
                    for output in tree.flatten(outputs):
                        if backend.is_tensor(output):
                            self.add_loss(self.activity_regularizer(output))

            # Set masks on outputs,
            # provided only the first positional input arg and its mask.
            # TODO: consider extending this to all args and kwargs.
            previous_mask = getattr(call_spec.first_arg, "_keras_mask", None)
            if self.supports_masking:
                self._set_mask_metadata(
                    call_spec.first_arg, outputs, previous_mask
                )
            elif previous_mask is not None:
                warnings.warn(
                    f"Layer '{self.name}' (of type {self.__class__.__name__}) "
                    "was passed an input with a mask attached to it. "
                    "However, this layer does not support masking and will "
                    "therefore destroy the mask information. Downstream "
                    "layers will not see the mask."
                )
        finally:
            # Destroy call context if we created it
            self._maybe_reset_call_context()
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    @traceback_utils.filter_traceback
    def stateless_call(
        self,
        trainable_variables,
        non_trainable_variables,
        *args,
        return_losses=False,
        **kwargs,
    ):
        """Call the layer without any side effects.

        Args:
            trainable_variables: List of trainable variables of the model.
            non_trainable_variables: List of non-trainable variables of the
                model.
            *args: Positional argumets to be passed to `call()`.
            return_losses: If `True`, `stateless_call()` will return the list of
                losses created during `call()` as part of its return values.
            **kwargs: Keyword arguments to be passed to `call()`.

        Returns:
            A tuple. By default, returns `(outputs, non_trainable_variables)`.
                If `return_losses = True`, then returns
                `(outputs, non_trainable_variables, losses)`.

        Note: `non_trainable_variables` include not only non-trainable weights
        such as `BatchNormalization` statistics, but also RNG seed state
        (if there are any random operations part of the layer, such as dropout),
        and `Metric` state (if there are any metrics attached to the layer).
        These are all elements of state of the layer.

        Example:

        ```python
        model = ...
        data = ...
        trainable_variables = model.trainable_variables
        non_trainable_variables = model.non_trainable_variables
        # Call the model with zero side effects
        outputs, non_trainable_variables = model.stateless_call(
            trainable_variables,
            non_trainable_variables,
            data,
        )
        # Attach the updated state to the model
        # (until you do this, the model is still in its pre-call state).
        for ref_var, value in zip(
            model.non_trainable_variables, non_trainable_variables
        ):
            ref_var.assign(value)
        ```
        """
        self._check_super_called()

        if not self.built:
            raise ValueError(
                f"To call stateless_call, {self.__class__.__name__} must be "
                "built (i.e. its variables must have been already created). "
                "You can build it by calling it on some data."
            )
        if len(trainable_variables) != len(self.trainable_variables):
            raise ValueError(
                "Argument `trainable_variables` must be a list of tensors "
                "corresponding 1:1 to "
                f"{self.__class__.__name__}().trainable_variables. "
                f"Received list with length {len(trainable_variables)}, "
                f"but expected {len(self.trainable_variables)} variables."
            )
        if len(non_trainable_variables) != len(self.non_trainable_variables):
            raise ValueError(
                "Argument `non_trainable_variables` must be a list of tensors "
                "corresponding 1:1 to "
                f"{self.__class__.__name__}().non_trainable_variables. "
                f"Received list with length {len(non_trainable_variables)}, "
                f"but expected {len(self.non_trainable_variables)} variables."
            )

        # Gather variable mapping
        trainable_mapping = zip(self.trainable_variables, trainable_variables)
        non_trainable_mapping = zip(
            self.non_trainable_variables, non_trainable_variables
        )
        mapping = list(trainable_mapping) + list(non_trainable_mapping)

        # Call in stateless scope
        losses = None
        with backend.StatelessScope(
            state_mapping=mapping, collect_losses=return_losses
        ) as scope:
            outputs = self.call(*args, **kwargs)
            if return_losses:
                losses = self.losses

        # Gather updated non-trainable variables
        non_trainable_variables = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                non_trainable_variables.append(new_v)
            else:
                non_trainable_variables.append(v)

        if return_losses:
            return outputs, non_trainable_variables, losses
        return outputs, non_trainable_variables

    def compute_output_spec(self, *args, **kwargs):
        if utils.is_default(self.compute_output_shape):
            return super().compute_output_spec(*args, **kwargs)
        else:
            # Use compute_output_shape() to return the right output spec
            call_spec = CallSpec(self._call_signature, args, kwargs)
            shapes_dict = get_shapes_dict(call_spec)
            shapes_dict = update_shapes_dict_for_target_fn(
                self.compute_output_shape,
                shapes_dict=shapes_dict,
                call_spec=call_spec,
                class_name=self.__class__.__name__,
            )
            output_shape = self.compute_output_shape(**shapes_dict)

            if (
                isinstance(output_shape, list)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                output_shape = tuple(output_shape)
            if not isinstance(output_shape, (list, tuple, dict)):
                try:
                    output_shape = tuple(output_shape)
                except:
                    raise ValueError(
                        "Method `compute_output_shape()` of layer "
                        f"{self.__class__.__name__} is returning "
                        "a type that cannot be interpreted as a shape. "
                        "It should return a shape tuple. "
                        f"Received: {output_shape}"
                    )
            if (
                isinstance(output_shape, tuple)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                return KerasTensor(output_shape, dtype=self.compute_dtype)
            # Case: nested. Could be a tuple/list of shapes, or a dict of
            # shapes. Could be deeply nested.
            return map_shape_structure(
                lambda s: KerasTensor(s, dtype=self.compute_dtype), output_shape
            )

    @utils.default
    def compute_output_shape(self, *args, **kwargs):
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} should implement "
            "`def compute_output_shape(self, input_shape)`."
        )

    def add_loss(self, loss):
        """Can be called inside of the `call()` method to add a scalar loss.

        Example:

        ```python
        class MyLayer(Layer):
            ...
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x
        ```
        """
        # Eager only.
        losses = tree.flatten(loss)
        for x in losses:
            if not backend.is_tensor(x):
                raise ValueError(
                    "`add_loss()` can only be called from inside `build()` or "
                    f"`call()`, on a tensor input. Received invalid value: {x}"
                )
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in losses:
                    scope.add_loss(loss)
                    self._loss_ids.add(id(loss))
        else:
            self._losses.extend(losses)

    def _get_own_losses(self):
        if backend.in_stateless_scope():
            losses = []
            scope = backend.get_stateless_scope()
            for loss in scope.losses:
                if id(loss) in self._loss_ids:
                    losses.append(loss)
            return losses
        else:
            return self._losses[:]

    def _get_regularization_losses(self):
        weight_regularization_losses = []
        for v in self.trainable_weights:
            regularizer = getattr(v, "regularizer", None)
            if regularizer is None:
                continue
            if backend.in_stateless_scope():
                v = backend.get_stateless_scope().get_current_value(v)
            weight_regularization_losses.append(regularizer(v))
        return weight_regularization_losses

    @property
    def losses(self):
        """List of scalar losses from `add_loss`, regularizers and sublayers."""
        losses = self._get_own_losses()
        for layer in self._flatten_layers(include_self=False):
            losses.extend(layer._get_own_losses())
        weight_regularization_losses = self._get_regularization_losses()
        losses.extend(weight_regularization_losses)
        return losses

    def _clear_losses(self):
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in scope.losses:
                    if id(x) in self._loss_ids:
                        scope.losses.remove(x)
        self._losses.clear()
        self._loss_ids.clear()
        for layer in self._layers:
            layer._clear_losses()

    def save_own_variables(self, store):
        """Saves the state of the layer.

        You can override this method to take full control of how the state of
        the layer is saved upon calling `model.save()`.

        Args:
            store: Dict where the state of the model will be saved.
        """
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            store[f"{i}"] = v.numpy()

    def load_own_variables(self, store):
        """Loads the state of the layer.

        You can override this method to take full control of how the state of
        the layer is loaded upon calling `keras.models.load_model()`.

        Args:
            store: Dict from which the state of the model will be loaded.
        """
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
        for i, v in enumerate(all_vars):
            v.assign(store[f"{i}"])

    def _track_variable(self, variable):
        if variable.trainable:
            self._tracker.add_to_store("trainable_variables", variable)
        else:
            self._tracker.add_to_store("non_trainable_variables", variable)

    def add_metric(self):
        # Permanently disabled
        raise NotImplementedError

    def count_params(self):
        """Count the total number of scalars composing the weights.

        Returns:
            An integer count.
        """
        if not self.built:
            raise ValueError(
                "You tried to call `count_params` "
                f"on layer '{self.name}', "
                "but the layer isn't built. "
                "You can build it manually via: "
                f"`layer.build(input_shape)`."
            )
        return summary_utils.count_params(self.weights)

    def _maybe_build(self, call_spec):
        if self.built:
            return

        shapes_dict = get_shapes_dict(call_spec)
        first_shape = next(iter(shapes_dict.values()), None)

        # If the layer has a build method, call it with our input shapes.
        if not utils.is_default(self.build):
            shapes_dict = update_shapes_dict_for_target_fn(
                self.build,
                shapes_dict=shapes_dict,
                call_spec=call_spec,
                class_name=self.__class__.__name__,
            )
            self.build(**shapes_dict)
            # Check input spec again (after build, since self.input_spec
            # may have been updated
            self._assert_input_compatibility(call_spec.first_arg)
            return

        # Otherwise, attempt to build the layer by calling it on symbolic input.
        if might_have_unbuilt_state(self):
            try:
                backend.compute_output_spec(
                    self.call, **call_spec.arguments_dict
                )
            except Exception as e:
                if call_spec.eager:
                    # Will let the actual eager call do state-building
                    return
                warnings.warn(
                    f"Layer '{self.name}' looks like it has unbuilt state, but "
                    "Keras is not able to trace the layer `call()` in order to "
                    "build it automatically. Possible causes:\n"
                    "1. The `call()` method of your layer may be crashing. Try "
                    "to `__call__()` the layer eagerly on some test input "
                    "first to see if it works. "
                    "E.g. `x = np.random.random((3, 4)); y = layer(x)`\n"
                    "2. If the `call()` method is correct, then you may need "
                    "to implement the `def build(self, input_shape)` method on "
                    "your layer. It should create all variables used by the "
                    "layer (e.g. by calling `layer.build()` on all its "
                    "children layers).\n"
                    f"Exception encoutered: ''{e}''"
                )
        self.build(first_shape)

    def _build_by_run_for_single_pos_arg(self, input_shape):
        # Case: all inputs are in the first arg (possibly nested).
        input_tensors = map_shape_structure(
            lambda s: backend.KerasTensor(s), input_shape
        )
        try:
            backend.compute_output_spec(self.call, input_tensors)
            return True
        except:
            return False

    def _build_by_run_for_kwargs(self, shapes_dict):
        # Case: inputs were recorded as multiple keyword arguments.
        if all(is_shape_tuple(s) for s in shapes_dict.values()):
            # Case: all input keyword arguments were plain tensors.
            input_tensors = {
                # We strip the `_shape` suffix to recover kwarg names.
                utils.removesuffix(k, "_shape"): backend.KerasTensor(shape)
                for k, shape in shapes_dict.items()
            }
            try:
                backend.compute_output_spec(self.call, **input_tensors)
                return True
            except:
                return False
        else:
            # Not supported: nested input keyword arguments.
            return False

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name}, built={self.built}>"
        )

    def __str__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name}, built={self.built}>"
        )

    def __setattr__(self, name, value):
        # Track Variables, Layers, Metrics, SeedGenerators.
        name, value = self._setattr_hook(name, value)
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        elif name != "_tracker":
            self._initializer_tracker()
        return super().__setattr__(name, value)

    def _check_super_called(self):
        if getattr(self, "_lock", True):
            raise RuntimeError(
                f"In layer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. Go add it!"
            )

    def _assert_input_compatibility(self, arg_0):
        if self.input_spec:
            input_spec.assert_input_compatibility(
                self.input_spec, arg_0, layer_name=self.name
            )

    def _get_call_context(self):
        """Returns currently active `CallContext`."""
        layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if layer_call_ctx is None:
            # Enter new call context.
            layer_call_ctx = CallContext(entry_layer=self)
            global_state.set_global_attribute(
                "current_call_ctx", layer_call_ctx
            )
            self._clear_losses()
        return layer_call_ctx

    def _maybe_reset_call_context(self):
        layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if layer_call_ctx is None or layer_call_ctx.entry_layer == self:
            global_state.set_global_attribute("current_call_ctx", None)

    def _flatten_layers(self, include_self=True, recursive=True):
        layers = []
        if include_self:
            layers.append(self)
        seen_object_ids = set()
        deque = collections.deque(self._layers)
        while deque:
            layer = deque.popleft()
            if id(layer) in seen_object_ids:
                continue
            seen_object_ids.add(id(layer))
            layers.append(layer)
            # Introspect recursively through sublayers.
            if recursive:
                deque.extendleft(layer._layers)
        return layers

    def _set_mask_metadata(self, inputs, outputs, previous_mask):
        flat_outputs = tree.flatten(outputs)

        mask_already_computed = all(
            getattr(x, "_keras_mask", None) is not None for x in flat_outputs
        )
        if mask_already_computed:
            return

        output_masks = self.compute_mask(inputs, previous_mask)
        if output_masks is None:
            return

        flat_masks = tree.flatten(output_masks)
        for tensor, mask in zip(flat_outputs, flat_masks):
            if getattr(tensor, "_keras_mask", None) is None:
                try:
                    # Numpy backend does not support masking.
                    if backend.backend() == "numpy":
                        warnings.warn(
                            "The NumPy backend does not support masking at this"
                            "time. Masks will be ignored."
                        )
                    tensor._keras_mask = mask
                except AttributeError:
                    # It's a C type.
                    pass

    @python_utils.default
    def get_config(self):
        self._check_super_called()
        base_config = super().get_config()
        config = {
            "trainable": self.trainable,
            "dtype": self.dtype_policy.name,
        }
        return {**base_config, **config}
```


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


## <a id="initializers-in-keras"></a>Initializers in Keras

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

Excerpt from `keras/src/initializers/constant_initializers.py`:
```python
@keras_export(["keras.initializers.Constant", "keras.initializers.constant"])
class Constant(Initializer):
    """Initializer that generates tensors with constant values.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Constant(10.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Constant(10.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        value: A Python scalar.
    """

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, shape, dtype=None):
        dtype = standardize_dtype(dtype)
        return ops.cast(self.value, dtype=dtype) * ops.ones(
            shape=shape, dtype=dtype
        )

    def get_config(self):
        return {"value": serialization_lib.serialize_keras_object(self.value)}

    @classmethod
    def from_config(cls, config):
        value = serialization_lib.deserialize_keras_object(config["value"])
        return cls(value)


@keras_export(["keras.initializers.Zeros", "keras.initializers.zeros"])
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Zeros()
    >>> layer = Dense(units=3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        dtype = standardize_dtype(dtype)
        return ops.zeros(shape, dtype=dtype)


@keras_export(["keras.initializers.Ones", "keras.initializers.ones"])
class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Also available via the shortcut function `ones`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Ones()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Ones()
    >>> layer = Dense(3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        dtype = standardize_dtype(dtype)
        return ops.ones(shape, dtype=dtype)


@keras_export(
    [
        "keras.initializers.IdentityInitializer",
        "keras.initializers.Identity",
        "keras.initializers.identity",
    ]
)
class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Only usable for generating 2D matrices.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Identity()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
        """
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices. "
                f"Received: shape={shape} of rank {len(shape)}."
            )
        dtype = standardize_dtype(dtype)
        return self.gain * ops.eye(*shape, dtype=dtype)
```

Excerpt from `keras/src/initializers/random_initializers.py`:
```python
@keras_export(
    [
        "keras.initializers.RandomNormal",
        "keras.initializers.random_normal",
    ]
)
class RandomNormal(Initializer):
    """Random normal initializer.

    Draws samples from a normal distribution for given parameters.

    Examples:

    >>> # Standalone usage:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self._init_seed = seed
        self.seed = seed or random.make_default_seed()
        super().__init__()

    def __call__(self, shape, dtype=None):
        return random.normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {"mean": self.mean, "stddev": self.stddev, "seed": seed_config}


@keras_export(
    [
        "keras.initializers.TruncatedNormal",
        "keras.initializers.truncated_normal",
    ]
)
class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    The values generated are similar to values from a
    `RandomNormal` initializer, except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Examples:

    >>> # Standalone usage:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self._init_seed = seed
        self.seed = seed or random.make_default_seed()
        super().__init__()

    def __call__(self, shape, dtype=None):
        return random.truncated_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {"mean": self.mean, "stddev": self.stddev, "seed": seed_config}


@keras_export(
    [
        "keras.initializers.RandomUniform",
        "keras.initializers.random_uniform",
    ]
)
class RandomUniform(Initializer):
    """Random uniform initializer.

    Draws samples from a uniform distribution for given parameters.

    Examples:

    >>> # Standalone usage:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        minval: A python scalar or a scalar keras tensor. Lower bound of the
            range of random values to generate (inclusive).
        maxval: A python scalar or a scalar keras tensor. Upper bound of the
            range of random values to generate (exclusive).
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self._init_seed = seed
        self.seed = seed or random.make_default_seed()
        super().__init__()

    def __call__(self, shape, dtype=None):
        return random.uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {
            "minval": self.minval,
            "maxval": self.maxval,
            "seed": seed_config,
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
```
