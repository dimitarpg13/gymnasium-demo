# Keras Library Internals

## <a id="contents"></a>Contents
[The Function class](#the-function-class)

[The Node class](#the-node-class)

[Keras Operation](#keras-operation)

[The Layers in Keras](#the-layers-in-keras)

[Initializers in Keras](#initializers-in-keras)

[The Keras Trainer class](#keras-trainer-class)

[The Keras TensorFlowTrainer class](#keras-tensorflow-trainer-class)

[The Keras Model class](#keras-model-class)

## <a id="the-function-class"></a>The Function Class

Excerpt from `keras/src/ops/function.py`
```python
@keras_export("keras.Function")
class Function(Operation):
    """Class that encapsulates a computation graph of Keras operations.

    You can use a `Function` to capture the computation graph linking
    some input tensors to some output tensors, and reapply the same
    computation on new inputs.

    A `Function` is similar to a Functional Model, with the difference
    that it is stateless (it does not track state variables)
    and does not implement the `Layer` API.

    Example:

    ```python
    input_1 = keras.KerasTensor(shape=(None, 2, 3))
    input_2 = keras.KerasTensor(shape=(None, 2, 3))
    x = input_1 + input_2
    output = keras.ops.sigmoid(x)
    fn = keras.Function(inputs=[input_1, input_2], outputs=output)

    input_1_val = np.random.random((4, 2, 3))
    input_2_val = np.random.random((4, 2, 3))
    output_val = fn([input_1_val, input_2_val])
    ```

    Args:
        inputs: `KerasTensor` instance or nested structured of
            `KerasTensor` instances.
        outputs: `KerasTensor` instance or nested structured of
            `KerasTensor` instances. They should be computable
            given only the values of `inputs`.
        name: String. The name of the function.
    """

    def __init__(self, inputs, outputs, name=None):
        super().__init__(name=name)

        if backend() == "tensorflow":
            # Temporary work around for
            # https://github.com/keras-team/keras/issues/931
            # This stop tensorflow from wrapping tf.function output in a
            # _DictWrapper object.
            _self_setattr_tracking = getattr(
                self, "_self_setattr_tracking", True
            )
            self._self_setattr_tracking = False
        self._inputs_struct = tree.map_structure(lambda x: x, inputs)
        self._outputs_struct = tree.map_structure(lambda x: x, outputs)
        self._inputs = tree.flatten(inputs)
        self._outputs = tree.flatten(outputs)
        if not self._inputs:
            raise ValueError(
                "`inputs` argument cannot be empty. Received:\n"
                f"inputs={inputs}\n"
                f"outputs={outputs}"
            )
        if not self._outputs:
            raise ValueError(
                "`outputs` argument cannot be empty. Received:\n"
                f"inputs={inputs}\n"
                f"outputs={outputs}"
            )

        if backend() == "tensorflow":
            self._self_setattr_tracking = _self_setattr_tracking

        (nodes, nodes_by_depth, operations, operations_by_depth) = map_graph(
            self._inputs, self._outputs
        )
        self._nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._operations = operations
        self._operations_by_depth = operations_by_depth

    @property
    def operations(self):
        return self._operations[:]

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def compute_output_spec(self, inputs):
        self._assert_input_compatibility(inputs)
        # Check if input shapes are identical to ref input shapes,
        # if so take a shortcut.
        shortcut = True
        for x, x_ref in zip(tree.flatten(inputs), self._inputs):
            if x.shape != x_ref.shape:
                shortcut = False
                break
        if shortcut:
            return tree.map_structure(
                lambda x: KerasTensor(shape=x.shape, dtype=x.dtype),
                self._outputs_struct,
            )
        # No luck; take the long road through the graph.
        # Original Keras used a cache to avoid recomputing all this
        # when known input shapes where seen again. Perhaps a good
        # idea to bring that back.
        return self._run_through_graph(
            inputs, operation_fn=lambda op: op.compute_output_spec
        )

    def call(self, inputs):
        """Computes output tensors for new inputs."""
        self._assert_input_compatibility(inputs)
        return self._run_through_graph(inputs, operation_fn=lambda op: op)

    def _run_through_graph(self, inputs, operation_fn):
        """Execute the graph.

        At each node we compute outputs via
        `operation_fn(node.operation)(*args, **kwargs)`.
        """
        inputs = tree.flatten(inputs)

        # Dictionary mapping reference tensors to computed tensors.
        tensor_dict = {}
        for x, y in zip(self.inputs, inputs):
            tensor_dict[id(x)] = y

        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:
                if not node.operation or node.is_input:
                    continue  # Input tensors already exist.

                if any(id(x) not in tensor_dict for x in node.input_tensors):
                    continue  # Node is not computable, try skipping.

                args, kwargs = node.arguments.fill_in(tensor_dict)
                outputs = operation_fn(node.operation)(*args, **kwargs)

                # Update tensor_dict.
                for x, y in zip(node.outputs, tree.flatten(outputs)):
                    tensor_dict[id(x)] = y

        output_tensors = []
        for x in self.outputs:
            output_tensors.append(tensor_dict[id(x)])

        return pack_sequence_as(self._outputs_struct, output_tensors)

    def _assert_input_compatibility(self, inputs):
        try:
            tree.assert_same_structure(
                inputs, self._inputs_struct, check_types=False
            )
        except ValueError:
            raise ValueError(
                "Function was called with an invalid input structure. "
                f"Expected input structure: {self._inputs_struct}\n"
                f"Received input structure: {inputs}"
            )
        for x, x_ref in zip(tree.flatten(inputs), self._inputs):
            if len(x.shape) != len(x_ref.shape):
                raise ValueError(
                    f"{self.__class__.__name__} was passed "
                    f"incompatible inputs. For input '{x_ref.name}', "
                    f"expected shape {x_ref.shape}, but received "
                    f"instead a tensor with shape {x.shape}."
                )
            for dim, ref_dim in zip(x.shape, x_ref.shape):
                if ref_dim is not None and dim is not None:
                    if dim != ref_dim:
                        raise ValueError(
                            f"{self.__class__.__name__} was passed "
                            f"incompatible inputs. For input '{x_ref.name}', "
                            f"expected shape {x_ref.shape}, but received "
                            f"instead a tensor with shape {x.shape}."
                        )


def make_node_key(op, node_index):
    return str(id(op)) + "_ib-" + str(node_index)


def map_graph(inputs, outputs):
    """Validates a graph's topology and gather its operations and nodes.

    Args:
        inputs: List of input tensors.
        outputs: List of outputs tensors.

    Returns:
        A tuple `(nodes, nodes_by_depth, operations, operations_by_depth)`.
        - network_nodes: dict mapping unique node keys to the Node instances
        - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
        - operations: list of Operation instances.
        - operations_by_depth: dict mapping ints (depth) to lists of Operation
            instances.
    """
    # "depth" is number of operations between output Node and the Node.
    # Nodes are ordered from inputs -> outputs.
    nodes_in_decreasing_depth, operation_indices = _build_map(outputs)
    network_nodes = {
        make_node_key(node.operation, node.operation._inbound_nodes.index(node))
        for node in nodes_in_decreasing_depth
    }

    nodes_depths = {}  # dict {node: depth value}
    operations_depths = {}  # dict {operation: depth value}

    for node in reversed(nodes_in_decreasing_depth):
        # If the depth is not set, the node has no outbound nodes (depth 0).
        depth = nodes_depths.setdefault(node, 0)

        # Update the depth of the corresponding operation
        previous_depth = operations_depths.get(node.operation, 0)
        # If we've seen this operation before at a higher depth,
        # we should use that depth instead of the node depth.
        # This is necessary for shared operations that have inputs at different
        # depth levels in the graph.
        depth = max(depth, previous_depth)
        operations_depths[node.operation] = depth
        nodes_depths[node] = depth

        # Update the depth of inbound nodes.
        # The "depth" of a node is the max of the depths
        # of all nodes it is connected to + 1.
        for node_dep in node.parent_nodes:
            previous_depth = nodes_depths.get(node_dep, 0)
            nodes_depths[node_dep] = max(depth + 1, previous_depth)

    # Handle inputs that are not connected to outputs.
    # We do not error out here because the inputs may be used to compute losses
    # and metrics.
    for input_t in inputs:
        input_operation = input_t._keras_history[0]
        if input_operation and input_operation not in operations_depths:
            operations_depths[input_operation] = 0
            operation_indices[input_operation] = -1
            nodes_depths[input_operation._inbound_nodes[0]] = 0
            network_nodes.add(make_node_key(input_operation, 0))

    # Build a dict {depth: list of nodes with this depth}
    nodes_by_depth = collections.defaultdict(list)
    for node, depth in nodes_depths.items():
        nodes_by_depth[depth].append(node)

    # Build a dict {depth: list of operations with this depth}
    operations_by_depth = collections.defaultdict(list)
    for operation, depth in operations_depths.items():
        operations_by_depth[depth].append(operation)

    # Get sorted list of operation depths.
    depth_keys = list(operations_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Set self.operations ordered by depth.
    operations = []
    for depth in depth_keys:
        operations_for_depth = operations_by_depth[depth]
        # Network.operations needs to have a deterministic order:
        # here we order them by traversal order.
        operations_for_depth.sort(key=lambda x: operation_indices[x])
        operations.extend(operations_for_depth)

    # Get sorted list of node depths.
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Check that all tensors required are computable.
    # computable_tensors: all tensors in the graph
    # that can be computed from the inputs provided.
    computable_tensors = set()
    for x in inputs:
        computable_tensors.add(x)

    operations_with_complete_input = []  # To provide a better error msg.
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            for x in tree.flatten(node.input_tensors):
                if x not in computable_tensors:
                    operation = node.operation
                    raise ValueError(
                        "Graph disconnected: cannot find parent for "
                        f"tensor {x} at operation '{operation}'. "
                        "The following previous operations were accessed "
                        f"without issue: {operations_with_complete_input}"
                    )
                operations_with_complete_input.append(operation.name)

            for x in tree.flatten(node.outputs):
                computable_tensors.add(x)

    # Ensure name unicity, which will be crucial for serialization
    # (since serialized nodes refer to operations by their name).
    all_names = [operation.name for operation in operations]
    for name in all_names:
        if all_names.count(name) != 1:
            raise ValueError(
                f'The name "{name}" is used {all_names.count(name)} '
                "times in the model. All operation names should be unique."
            )
    return network_nodes, nodes_by_depth, operations, operations_by_depth


def _build_map(outputs):
    """Topologically sort nodes in order from inputs to outputs.

    It uses a depth-first search to topologically sort nodes that appear in the
    _keras_history connectivity metadata of `outputs`.

    Args:
        outputs: the output tensors whose _keras_history metadata should be
                walked. This may be an arbitrary nested structure.

    Returns:
        A tuple like (ordered_nodes, operation_to_first_traversal_index)
        ordered_nodes: list of nodes appearing in the keras history,
            topologically sorted from original inputs to the `outputs`.
            (If outputs have different sets of ancestors, the inputs to one
            output may appear after a different output).
        operation_to_first_traversal_index:
            A dict mapping operation to the traversal index in the DFS where it
            is seen. Note: if a operation is shared by several nodes, the dict
            will onlystore the index corresponding to the *first* time the
            operation seen.
    """
    finished_nodes = set()
    nodes_in_progress = set()
    nodes_in_decreasing_depth = []  # nodes from inputs -> outputs.
    operation_indices = {}  # operation -> in traversal order.
    for output in tree.flatten(outputs):
        _build_map_helper(
            output,
            finished_nodes,
            nodes_in_progress,
            nodes_in_decreasing_depth,
            operation_indices,
        )
    return nodes_in_decreasing_depth, operation_indices


def _build_map_helper(
    tensor,
    finished_nodes,
    nodes_in_progress,
    nodes_in_decreasing_depth,
    operation_indices,
):
    """Recursive helper for `_build_map`."""
    (
        operation,
        node_index,
        _,
    ) = tensor._keras_history
    if not operation:
        return

    node = operation._inbound_nodes[node_index]

    # Don't repeat work for shared subgraphs
    if node in finished_nodes:
        return

    # Prevent cycles.
    if node in nodes_in_progress:
        raise ValueError(
            f"Tensor {tensor} from operation '{operation.name}' is part of a "
            "cycle."
        )

    # Store the traversal order for operation sorting.
    if operation not in operation_indices:
        operation_indices[operation] = len(operation_indices)

    # Propagate to all previous tensors connected to this node.
    nodes_in_progress.add(node)
    if not node.is_input:
        for tensor in node.input_tensors:
            _build_map_helper(
                tensor,
                finished_nodes,
                nodes_in_progress,
                nodes_in_decreasing_depth,
                operation_indices,
            )

    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)
```
[Go back to the beginning of the Section](#the-function-class)


## <a id="keras-operation"></a>Keras Operation

Excerpt from `keras/src/ops/operation.py`:
```python
@keras_export("keras.Operation")
class Operation:
    def __init__(self, name=None):
        if name is None:
            name = auto_name(self.__class__.__name__)
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name} (of type {type(name)})"
            )
        self.name = name
        self._inbound_nodes = []
        self._outbound_nodes = []

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        if traceback_utils.is_traceback_filtering_enabled():
            # Wrap self.call to provide helpful info in case of exception
            if any_symbolic_tensors(args, kwargs):
                call_fn = self.symbolic_call
            else:
                call_fn = self.call
            call_fn = traceback_utils.inject_argument_info_in_traceback(
                call_fn,
                object_name=(f"{self.__class__.__name__}.call()"),
            )
            return call_fn(*args, **kwargs)

        # Plain flow.
        if any_symbolic_tensors(args, kwargs):
            return self.symbolic_call(*args, **kwargs)
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args, **kwargs):
        # Perform shape/dtype inference.
        outputs = self.compute_output_spec(*args, **kwargs)
        # Record a new node in the operations graph.
        # The Node wires itself to inbound and outbound ops.  The
        # Node constructor updates this op's self._inbound_nodes,
        # sets _keras_history on the outputs, and adds itself to the
        # `_outbound_nodes` of the ops that produced the inputs to this
        # call.
        Node(
            operation=self, call_args=args, call_kwargs=kwargs, outputs=outputs
        )
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def compute_output_spec(self, *args, **kwargs):
        try:
            return backend.compute_output_spec(self.call, *args, **kwargs)
        except Exception as e:
            if isinstance(e, TypeError):
                raise e
            else:
                new_e = RuntimeError(
                    "Could not automatically infer the output shape / dtype of "
                    f"'{self.name}' (of type {self.__class__.__name__}). "
                    f"Either the `{self.__class__.__name__}.call()` method "
                    f"is incorrect, or you need to implement the "
                    f"`{self.__class__.__name__}.compute_output_spec() / "
                    "compute_output_shape()` method. "
                    f"Error encountered:\n\n{e}"
                )
                raise new_e.with_traceback(e.__traceback__) from None

    def __new__(cls, *args, **kwargs):
        """We override __new__ to saving serializable constructor arguments.

        These arguments are used to auto-generate an object serialization
        config, which enables user-created subclasses to be serializable
        out of the box in most cases without forcing the user
        to manually implement `get_config()`.
        """
        # Generate a config to be returned by default by `get_config()`.
        arg_names = inspect.getfullargspec(cls.__init__).args
        kwargs.update(dict(zip(arg_names[1 : len(args) + 1], args)))
        instance = super(Operation, cls).__new__(cls)
        # For safety, we only rely on auto-configs for a small set of
        # serializable types.
        supported_types = (str, int, float, bool, type(None))
        try:
            flat_arg_values = tree.flatten(kwargs)
            auto_config = True
            for value in flat_arg_values:
                if not isinstance(value, supported_types):
                    auto_config = False
                    break
        except TypeError:
            auto_config = False
        try:
            instance._lock = False
            if auto_config:
                from keras.src.saving import serialization_lib

                instance._auto_config = serialization_lib.SerializableDict(
                    **kwargs
                )
            else:
                instance._auto_config = None
            instance._lock = True
        except RecursionError:
            # Setting an instance attribute in __new__ has the potential
            # to trigger an infinite recursion if a subclass overrides
            # setattr in an unsafe way.
            pass
        return instance

    @python_utils.default
    def get_config(self):
        """Returns the config of the object.

        An object config is a Python dictionary (serializable)
        containing the information needed to re-instantiate it.
        """
        config = {
            "name": self.name,
        }

        if not python_utils.is_default(self.get_config):
            # In this case the subclass implements get_config()
            return config

        # In this case the subclass doesn't implement get_config():
        # Let's see if we can autogenerate it.
        if getattr(self, "_auto_config", None) is not None:
            xtra_args = set(config.keys())
            config.update(self._auto_config.config)
            # Remove args non explicitly supported
            argspec = inspect.getfullargspec(self.__init__)
            if argspec.varkw != "kwargs":
                for key in xtra_args - xtra_args.intersection(argspec.args[1:]):
                    config.pop(key, None)
            return config
        else:
            raise NotImplementedError(
                textwrap.dedent(
                    f"""
        Object {self.__class__.__name__} was created by passing
        non-serializable argument values in `__init__()`,
        and therefore the object must override `get_config()` in
        order to be serializable. Please implement `get_config()`.

        Example:

        class CustomLayer(keras.layers.Layer):
            def __init__(self, arg1, arg2, **kwargs):
                super().__init__(**kwargs)
                self.arg1 = arg1
                self.arg2 = arg2

            def get_config(self):
                config = super().get_config()
                config.update({
                    "arg1": self.arg1,
                    "arg2": self.arg2,
                })
                return config"""
                )
            )

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        Args:
            config: A Python dictionary, typically the
                output of get_config.

        Returns:
            A layer instance.
        """
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when deserializing class '{cls.__name__}' using "
                f"config={config}.\n\nException encountered: {e}"
            )

    def __repr__(self):
        return f"<Operation name={self.name}>"

    @property
    def input(self):
        """Retrieves the input tensor(s) of a symbolic operation.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Input tensor or list of input tensors.
        """
        return self._get_node_attribute_at_index(0, "input_tensors", "input")

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Output tensor or list of output tensors.
        """
        return self._get_node_attribute_at_index(0, "output_tensors", "output")

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Private utility to retrieves an attribute (e.g. inputs) from a node.

        This is used to implement the properties:
        - output
        - input

        Args:
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        Returns:
            The operation's attribute `attr` at the node of index `node_index`.
        """
        if not self._inbound_nodes:
            raise ValueError(
                f"The layer {self.name} has never been called "
                f"and thus has no defined {attr_name}."
            )
        if not len(self._inbound_nodes) > node_index:
            raise ValueError(
                f"Asked to get {attr_name} at node "
                f"{node_index}, but the operation has only "
                f"{len(self._inbound_nodes)} inbound nodes."
            )
        values = getattr(self._inbound_nodes[node_index], attr)
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        else:
            return values

    # Hooks for backend layer classes
    def _post_build(self):
        """Can be overridden for per backend post build actions."""
        pass

    def _setattr_hook(self, name, value):
        """Can be overridden for per backend post build actions."""
        return name, value
```
[Go back to the beginning of the Section](#keras-operation)

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
[Go back to the beginning of the Section](#the-layers-in-keras)

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
[Go back to the beginning of the Section](#the-layers-in-keras)

## <a id="initializers-in-keras"></a>Initializers in Keras

[Go back to Contents](#contents)

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
[Go back to the beginning of the Section](#initializers-in-keras)

## <a id="keras-trainer-class"></a>Keras Trainer class

[Go back to Contents](#contents)

Excerpt from `keras/src/trainers/trainer.py`

```python
class Trainer:
    def __init__(self):
        self._lock = False
        self._run_eagerly = False
        self._jit_compile = None
        self.compiled = False
        self.loss = None
        self.steps_per_execution = 1

    @traceback_utils.filter_traceback
    @tracking.no_automatic_dependency_tracking
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    ):
        """Configures the model for training.

        Example:

        ```python
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.FalseNegatives(),
            ],
        )
        ```

        Args:
            optimizer: String (name of optimizer) or optimizer instance. See
                `keras.optimizers`.
            loss: Loss function. May be a string (name of loss function), or
                a `keras.losses.Loss` instance. See `keras.losses`. A
                loss function is any callable with the signature
                `loss = fn(y_true, y_pred)`, where `y_true` are the ground truth
                values, and `y_pred` are the model's predictions.
                `y_true` should have shape `(batch_size, d0, .. dN)`
                (except in the case of sparse loss functions such as
                sparse categorical crossentropy which expects integer arrays of
                shape `(batch_size, d0, .. dN-1)`).
                `y_pred` should have shape `(batch_size, d0, .. dN)`.
                The loss function should return a float tensor.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions of
                different model outputs. The loss value that will be minimized
                by the model will then be the *weighted sum* of all individual
                losses, weighted by the `loss_weights` coefficients.  If a list,
                it is expected to have a 1:1 mapping to the model's outputs. If
                a dict, it is expected to map output names (strings) to scalar
                coefficients.
            metrics: List of metrics to be evaluated by the model during
                training and testing. Each of this can be a string (name of a
                built-in function), function or a `keras.metrics.Metric`
                instance. See `keras.metrics`. Typically you will use
                `metrics=['accuracy']`. A function is any callable with the
                signature `result = fn(y_true, _pred)`. To specify different
                metrics for different outputs of a multi-output model, you could
                also pass a dictionary, such as
                `metrics={'a':'accuracy', 'b':['accuracy', 'mse']}`.
                You can also pass a list to specify a metric or a list of
                metrics for each output, such as
                `metrics=[['accuracy'], ['accuracy', 'mse']]`
                or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass
                the strings 'accuracy' or 'acc', we convert this to one of
                `keras.metrics.BinaryAccuracy`,
                `keras.metrics.CategoricalAccuracy`,
                `keras.metrics.SparseCategoricalAccuracy` based on the
                shapes of the targets and of the model output. A similar
                conversion is done for the strings `"crossentropy"`
                and `"ce"` as well.
                The metrics passed here are evaluated without sample weighting;
                if you would like sample weighting to apply, you can specify
                your metrics via the `weighted_metrics` argument instead.
            weighted_metrics: List of metrics to be evaluated and weighted by
                `sample_weight` or `class_weight` during training and testing.
            run_eagerly: Bool. If `True`, this model's forward pass
                 will never be compiled. It is recommended to leave this
                 as `False` when training (for best performance),
                 and to set it to `True` when debugging.
            steps_per_execution: Int. The number of batches to run
                during each a single compiled function call. Running multiple
                batches inside a single compiled function call can
                greatly improve performance on TPUs or small models with a large
                Python overhead. At most, one full epoch will be run each
                execution. If a number larger than the size of the epoch is
                passed, the execution will be truncated to the size of the
                epoch. Note that if `steps_per_execution` is set to `N`,
                `Callback.on_batch_begin` and `Callback.on_batch_end` methods
                will only be called every `N` batches (i.e. before/after
                each compiled function execution).
                Not supported with the PyTorch backend.
            jit_compile: Bool or `"auto"`. Whether to use XLA compilation when
                compiling a model. For `jax` and `tensorflow` backends,
                `jit_compile="auto"` enables XLA compilation if the model
                supports it, and disabled otherwise.
                For `torch` backend, `"auto"` will default to eager
                execution and `jit_compile=True` will run with `torch.compile`
                with the `"inductor"` backend.
            auto_scale_loss: Bool. If `True` and the model dtype policy is
                `"mixed_float16"`, the passed optimizer will be automatically
                wrapped in a `LossScaleOptimizer`, which will dynamically
                scale the loss to prevent underflow.
        """
        self.optimizer = optimizers.get(optimizer)
        if (
            auto_scale_loss
            and self.dtype_policy.name == "mixed_float16"
            and self.optimizer
            and not isinstance(self.optimizer, LossScaleOptimizer)
        ):
            self.optimizer = LossScaleOptimizer(
                self.optimizer, name="loss_scale_optimizer"
            )
        if hasattr(self, "output_names"):
            output_names = self.output_names
        else:
            output_names = None
        if loss is not None:
            self._compile_loss = CompileLoss(
                loss, loss_weights, output_names=output_names
            )
            self.loss = loss
        else:
            self._compile_loss = None
        if metrics is not None or weighted_metrics is not None:
            self._compile_metrics = CompileMetrics(
                metrics, weighted_metrics, output_names=output_names
            )
        else:
            self._compile_metrics = None
        if jit_compile == "auto":
            if run_eagerly:
                jit_compile = False
            else:
                jit_compile = self._resolve_auto_jit_compile()
        if jit_compile and run_eagerly:
            jit_compile = False
            warnings.warn(
                "If `run_eagerly` is True, then `jit_compile` "
                "cannot also be True. Disabling `jit_compile`.",
                stacklevel=2,
            )

        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self.compiled = True
        self._loss_tracker = metrics_module.Mean(name="loss")
        self.steps_per_execution = steps_per_execution

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        self._compile_config = serialization_lib.SerializableDict(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
        )

    @property
    def jit_compile(self):
        if self._jit_compile is None:
            # Value was never set. Resolve it now.
            self._jit_compile = self._resolve_auto_jit_compile()
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value):
        if value and not model_supports_jit(self):
            warnings.warn(
                "Model doesn't support `jit_compile=True`. "
                "Proceeding with `jit_compile=False`."
            )
            self._jit_compile = False
        else:
            self._jit_compile = value

    def _resolve_auto_jit_compile(self):
        if backend.backend() == "torch":
            # jit_compile = "auto" with the pytorch backend defaults to eager
            return False

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            devices = tf.config.list_physical_devices()
            if not list(filter(lambda x: x.device_type != "CPU", devices)):
                # Disable XLA on CPU-only machines.
                return False

            if self._distribute_strategy:
                # Disable XLA with tf.distribute
                return False

        if model_supports_jit(self):
            return True
        return False

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value):
        self._run_eagerly = value

    @property
    def metrics(self):
        metrics = [self._loss_tracker] if self.compiled else []
        metrics.extend(self._metrics[:])
        if self.compiled and self._compile_metrics is not None:
            metrics += [self._compile_metrics]
        return metrics

    @property
    def metrics_names(self):
        return [m.name for m in self.metrics]

    @property
    def metrics_variables(self):
        vars = []
        for metric in self.metrics:
            vars.extend(metric.variables)
        return vars

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        """Compute the total loss, validate it, and return it.

        Subclasses can optionally override this method to provide custom loss
        computation logic.

        Example:

        ```python
        class MyModel(Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_tracker = metrics.Mean(name='loss')

            def compute_loss(self, x, y, y_pred, sample_weight):
                loss = ops.means((y_pred - y) ** 2)
                loss += ops.sum(self.losses)
                self.loss_tracker.update_state(loss)
                return loss

            def reset_metrics(self):
                self.loss_tracker.reset_state()

            @property
            def metrics(self):
                return [self.loss_tracker]

        inputs = layers.Input(shape=(10,), name='my_input')
        outputs = layers.Dense(10)(inputs)
        model = MyModel(inputs, outputs)
        model.add_loss(ops.sum(outputs))

        optimizer = SGD()
        model.compile(optimizer, loss='mse', steps_per_execution=10)
        dataset = ...
        model.fit(dataset, epochs=2, steps_per_epoch=10)
        print(f"Custom loss: {model.loss_tracker.result()}")
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model(x)`)
            sample_weight: Sample weights for weighting the loss function.
            allow_empty: If `False`, the method will error out if
                no loss has been computed by the model. If `True`, then
                if no loss is computed, the method returns 0.

        Returns:
            The total loss as a scalar tensor, or `None` if no loss results
            (which is the case when called by `Model.test_step`).
        """
        del x  # The default implementation does not use `x`.
        losses = []
        if self._compile_loss is not None:
            loss = self._compile_loss(y, y_pred, sample_weight)
            if loss is not None:
                losses.append(loss)
        for loss in self.losses:
            losses.append(ops.cast(loss, dtype=backend.floatx()))
        if not allow_empty and len(losses) == 0:
            raise ValueError(
                "No loss to compute. Provide a `loss` argument in `compile()`."
            )
        if len(losses) == 1:
            total_loss = losses[0]
        elif len(losses) == 0:
            total_loss = ops.zeros(())
        else:
            total_loss = ops.sum(losses)
        return total_loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic.

        Example:

        ```python
        class MyModel(Sequential):
            def compute_metrics(self, x, y, y_pred, sample_weight):
                # This super call updates `self.compiled_metrics` and returns
                # results for all metrics listed in `self.metrics`.
                metric_results = super().compute_metrics(
                    x, y, y_pred, sample_weight)

                # Note that `self.custom_metric` is not listed
                # in `self.metrics`.
                self.custom_metric.update_state(x, y, y_pred, sample_weight)
                metric_results['metric_name'] = self.custom_metric.result()
                return metric_results
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model output of `model.call(x)`.
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            A `dict` containing values that will be passed to
            `keras.callbacks.CallbackList.on_train_batch_end()`. Typically,
            the values of the metrics listed in `self.metrics` are returned.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        del x  # The default implementation does not use `x`.
        if self._compile_metrics is not None:
            self._compile_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_metrics_result(self):
        """Returns the model's metrics values as a dict.

        If any of the metric result is a dict (containing multiple metrics),
        each of them gets added to the top level returned dict of this method.

        Returns:
            A `dict` containing values of the metrics listed in `self.metrics`.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return self._pythonify_logs(return_metrics)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        """Trains the model for a fixed number of epochs (dataset iterations).

        Args:
            x: Input data. It could be:
                - A NumPy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
                - A tensor, or a list of tensors
                (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
                - A `tf.data.Dataset`. Should return a tuple
                of either `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
                - A `keras.utils.PyDataset` returning `(inputs,
                targets)` or `(inputs, targets, sample_weights)`.
            y: Target data. Like the input data `x`,
                it could be either NumPy array(s) or backend-native tensor(s).
                If `x` is a dataset, generator,
                or `keras.utils.PyDataset` instance, `y` should
                not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.PyDataset`
                instances (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided
                (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                "auto" becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g., in a production environment). Defaults to `"auto"`.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `keras.callbacks`. Note
                `keras.callbacks.ProgbarLogger` and
                `keras.callbacks.History` callbacks are created
                automatically and need not be passed to `model.fit()`.
                `keras.callbacks.ProgbarLogger` is created
                or not based on the `verbose` argument in `model.fit()`.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This
                argument is not supported when `x` is a dataset, generator or
                `keras.utils.PyDataset` instance.
                If both `validation_data` and `validation_split` are provided,
                `validation_data` will override `validation_split`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data. Thus, note the fact
                that the validation loss of data provided using
                `validation_split` or `validation_data` is not affected by
                regularization layers like noise and dropout.
                `validation_data` will override `validation_split`.
                It could be:
                - A tuple `(x_val, y_val)` of NumPy arrays or tensors.
                - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
                arrays.
                - A `tf.data.Dataset`.
                - A Python generator or `keras.utils.PyDataset` returning
                `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            shuffle: Boolean, whether to shuffle the training data
                before each epoch. This argument is
                ignored when `x` is a generator or a `tf.data.Dataset`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must be
                one-hot encoded, or an explicit final dimension of `1` must
                be included for sparse class labels.
            sample_weight: Optional NumPy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                NumPy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                This argument is not supported when `x` is a dataset, generator,
                or `keras.utils.PyDataset` instance, instead provide the
                sample_weights as the third element of `x`.
                Note that sample weighting does not apply to metrics specified
                via the `metrics` argument in `compile()`. To apply sample
                weighting to your metrics, you can specify them via the
                `weighted_metrics` in `compile()` instead.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                backend-native tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If `x` is a
                `tf.data.Dataset`, and `steps_per_epoch`
                is `None`, the epoch will run until the input dataset is
                exhausted.  When passing an infinitely repeating dataset, you
                must specify the `steps_per_epoch` argument. If
                `steps_per_epoch=-1` the training will run indefinitely with an
                infinitely repeating dataset.
            validation_steps: Only relevant if `validation_data` is provided.
                Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If `validation_steps` is `None`,
                validation will run until the `validation_data` dataset is
                exhausted. In the case of an infinitely repeated dataset, it
                will run into an infinite loop. If `validation_steps` is
                specified and only part of the dataset will be consumed, the
                evaluation will start from the beginning of the dataset at each
                epoch. This ensures that the same validation samples are used
                every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in
                the form of datasets or `keras.utils.PyDataset`
                instances (since they generate batches).
            validation_freq: Only relevant if validation data is provided.
                Specifies how many training epochs to run
                before a new validation run is performed,
                e.g. `validation_freq=2` runs validation every 2 epochs.

        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass an iterator like object such as a
            `tf.data.Dataset` or a `keras.utils.PyDataset` to `fit()`,
            which will in fact yield not only features (`x`)
            but optionally targets (`y`) and sample weights (`sample_weight`).
            Keras requires that the output of such iterator-likes be
            unambiguous. The iterator should return a tuple
            of length 1, 2, or 3, where the optional second and third elements
            will be used for `y` and `sample_weight` respectively.
            Any other type provided will be wrapped in
            a length-one tuple, effectively treating everything as `x`. When
            yielding dicts, they should still adhere to the top-level tuple
            structure,
            e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
            features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the `namedtuple`. The reason is
            that it behaves like both an ordered datatype (tuple) and a mapping
            datatype (dict). So given a namedtuple of the form:
            `namedtuple("example_tuple", ["y", "x"])`
            it is ambiguous whether to reverse the order of the elements when
            interpreting the value. Even worse is a tuple of the form:
            `namedtuple("other_tuple", ["x", "y", "z"])`
            where it is unclear if the tuple was intended to be unpacked
            into `x`, `y`, and `sample_weight` or passed through
            as a single element to `x`.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        raise NotImplementedError

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches (see the `batch_size` arg.)

        Args:
            x: Input data. It could be:
                - A NumPy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data.Dataset`. Should return a tuple
                    of either `(inputs, targets)` or
                    `(inputs, targets, sample_weights)`.
                - A generator or `keras.utils.PyDataset` returning
                    `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            y: Target data. Like the input data `x`, it could be either NumPy
                array(s) or backend-native tensor(s).
                If `x` is a `tf.data.Dataset` or `keras.utils.PyDataset`
                instance, `y` should not be specified
                (since targets will be obtained from the iterator/dataset).
            batch_size: Integer or `None`. Number of samples per batch of
                computation. If unspecified, `batch_size` will default to 32. Do
                not specify the `batch_size` if your data is in the form of a
                dataset, generators, or `keras.utils.PyDataset` instances
                (since they generate batches).
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file, so `verbose=2` is
                recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            sample_weight: Optional NumPy array of weights for the test samples,
                used for weighting the loss function. You can either pass a flat
                (1D) NumPy array with the same length as the input samples
                (1:1 mapping between weights and samples), or in the case of
                temporal data, you can pass a 2D array with shape `(samples,
                sequence_length)`, to apply a different weight to every
                timestep of every sample. This argument is not supported when
                `x` is a dataset, instead pass sample weights as the third
                element of `x`.
            steps: Integer or `None`. Total number of steps (batches of samples)
                before declaring the evaluation round finished. Ignored with the
                default value of `None`. If `x` is a `tf.data.Dataset` and
                `steps` is `None`, evaluation will run until the dataset
                is exhausted.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric.
                If `False`, they are returned as a list.

        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        raise NotImplementedError

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        """Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for batch
        processing of large numbers of inputs. It is not intended for use inside
        of loops that iterate over your data and process small numbers of inputs
        at a time.

        For small numbers of inputs that fit in one batch,
        directly use `__call__()` for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `BatchNormalization` that behave differently during
        inference.

        Note: See [this FAQ entry](
        https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
        for more details about the difference between `Model` methods
        `predict()` and `__call__()`.

        Args:
            x: Input samples. It could be:
                - A NumPy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A `tf.data.Dataset`.
                - A `keras.utils.PyDataset` instance.
            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras.utils.PyDataset`
                instances (since they generate batches).
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
                If `x` is a `tf.data.Dataset` and `steps` is `None`,
                `predict()` will run until the input dataset is exhausted.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        """Runs a single gradient update on a single batch of data.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) to apply to the model's loss for the samples
                from this class during training. This can be useful to tell the
                model to "pay more attention" to samples from an
                under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must
                be one-hot encoded, or an explicit final dimension of 1
                must be included for sparse class labels.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        raise NotImplementedError

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        """Test the model on a single batch of samples.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        raise NotImplementedError

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Args:
            x: Input data. It must be array-like.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def get_compile_config(self):
        """Returns a serialized config with information for compiling the model.

        This method returns a config dictionary containing all the information
        (optimizer, loss, metrics, etc.) with which the model was compiled.

        Returns:
            A dict containing information for compiling the model.
        """
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()

    def compile_from_config(self, config):
        """Compiles the model with the information given in config.

        This method uses the information in the config (optimizer, loss,
        metrics, etc.) to compile the model.

        Args:
            config: Dict containing information for compiling the model.
        """
        has_overridden_compile = self.__class__.compile != Trainer.compile
        if has_overridden_compile:
            warnings.warn(
                "`compile()` was not called as part of model loading "
                "because the model's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading.",
                stacklevel=2,
            )
            return
        config = serialization_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )

    def _pythonify_logs(self, logs):
        result = {}
        for key, value in sorted(logs.items()):
            if isinstance(value, dict):
                result.update(self._pythonify_logs(value))
            else:
                try:
                    value = float(value)
                except:
                    pass
                result[key] = value
        return result

    def _flatten_metrics_in_order(self, logs):
        """Turns `logs` dict into a list as per key order of `metrics_names`."""
        metric_names = [m.name for m in self.metrics]
        results = []
        for name in metric_names:
            if name in logs:
                results.append(logs[name])
        for key in sorted(logs.keys()):
            if key not in metric_names:
                results.append(logs[key])
        if len(results) == 1:
            return results[0]
        return results

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the model."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)

    def _symbolic_build(self, iterator=None, data_batch=None):
        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        optimizer_unbuilt = (
            self.optimizer is not None and not self.optimizer.built
        )
        if model_unbuilt or compile_metrics_unbuilt or optimizer_unbuilt:
            if data_batch is None:
                for _, data in iterator.enumerate_epoch():
                    data_batch = data[0]
                    break

        if model_unbuilt or compile_metrics_unbuilt:
            # Create symbolic tensors matching an input batch.

            def to_symbolic_input(v):
                if v is None:
                    return None
                return backend.KerasTensor(
                    v.shape, backend.standardize_dtype(v.dtype)
                )

            data_batch = tree.map_structure(to_symbolic_input, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            # Build all model state with `backend.compute_output_spec`.
            try:
                y_pred = backend.compute_output_spec(self, x)
            except Exception as e:
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it.\n"
                    "Exception encountered:\n"
                    f"'{e}'"
                )
            if compile_metrics_unbuilt:
                # Build all metric state with `backend.compute_output_spec`.
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
        if optimizer_unbuilt:
            # Build optimizer
            self.optimizer.build(self.trainable_variables)
        self._post_build()
```
[Go back to the beginning of the Section](#keras-trainer-class)

## <a id="keras-tensorflow-trainer-class"></a>Keras TensorFlowTrainer class

[Go back to Contents](#contents)

Excerpt from `keras/src/backend/tensorflow/trainer.py`

```python
class TensorFlowTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Model must be created under scope of DistStrat it will be trained
        # with.
        if tf.distribute.has_strategy():
            self._distribute_strategy = tf.distribute.get_strategy()
        else:
            self._distribute_strategy = None

        self._distribute_reduction_method = None
        self._supports_reduce_retracing = Version(tf.__version__) >= Version(
            "2.9.0"
        )

    @property
    def distribute_strategy(self):
        return self._distribute_strategy or tf.distribute.get_strategy()

    @property
    def distribute_reduction_method(self):
        return self._distribute_reduction_method or "auto"

    @distribute_reduction_method.setter
    def distribute_reduction_method(self, value):
        self._distribute_reduction_method = value

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )
            self._loss_tracker.update_state(loss)
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            trainable_weights = self.trainable_weights
            gradients = tape.gradient(loss, trainable_weights)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self.compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
        )
        self._loss_tracker.update_state(loss)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            return self.train_step(data)

        if not self.run_eagerly:
            kwargs = {"jit_compile": self.jit_compile}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single training step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution):
                outputs = one_step_on_iterator(iterator)
            return outputs

        if self.steps_per_execution > 1:
            train_function = multi_step_on_iterator
        else:
            train_function = one_step_on_iterator

        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            train_function = tf.function(train_function, **kwargs)

        self.train_function = train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            return self.test_step(data)

        if not self.run_eagerly and self.jit_compile:
            kwargs = {"jit_compile": True}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single test step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution):
                outputs = one_step_on_iterator(iterator)
            return outputs

        if self.steps_per_execution > 1:
            test_function = multi_step_on_iterator
        else:
            test_function = one_step_on_iterator

        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            test_function = tf.function(test_function, **kwargs)

        self.test_function = test_function

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            return self.predict_step(data)

        if not self.run_eagerly and self.jit_compile:
            kwargs = {"jit_compile": True}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data_distributed(data):
            data = data[0]
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_data(data):
            outputs = one_step_on_data_distributed(data[:1])
            for single_step_data in data[1:]:
                step_outputs = one_step_on_data_distributed([single_step_data])
                outputs = tf.nest.map_structure(
                    lambda t1, t2: concat([t1, t2]), outputs, step_outputs
                )
            return outputs

        if self.steps_per_execution > 1:
            predict_function = multi_step_on_data
        else:
            predict_function = one_step_on_data_distributed

        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})

            predict_function = tf.function(predict_function, **kwargs)

        self.predict_function = predict_function

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        self._assert_compile_called("fit")
        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter_utils.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TFEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            distribute_strategy=self.distribute_strategy,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = None
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for step, iterator in epoch_iterator.enumerate_epoch():
                    callbacks.on_train_batch_begin(step)
                    logs = self.train_function(iterator)
                    callbacks.on_train_batch_end(
                        step, self._pythonify_logs(logs)
                    )
                    if self.stop_training:
                        break

            # Override with model metrics instead of last step logs
            epoch_logs = self.get_metrics_result()

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TFEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        distribute_strategy=self.distribute_strategy,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        self._assert_compile_called("evaluate")
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TFEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                distribute_strategy=self.distribute_strategy,
                steps_per_execution=self.steps_per_execution,
            )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_test_batch_begin(step)
                logs = self.test_function(iterator)
                callbacks.on_test_batch_end(step, self._pythonify_logs(logs))
                if self.stop_evaluating:
                    break
        logs = self.get_metrics_result()
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = TFEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            distribute_strategy=self.distribute_strategy,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tf.nest.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        def get_data(iterator):
            """Returns data for the next execution."""
            data = []
            for _ in range(self.steps_per_execution):
                try:
                    single_step_data = next(iterator)
                except (StopIteration, tf.errors.OutOfRangeError) as e:
                    if hasattr(data, "__len__") and len(data) > 0:
                        # Suppress the error when still have remaining data.
                        return data
                    else:
                        # Re-raise the error for
                        # TFEpochIterator.catch_stop_iteration() to catch when
                        # no data left.
                        raise e
                data.append(single_step_data)
            return data

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_predict_batch_begin(step)
                data = get_data(iterator)
                batch_outputs = self.predict_function(data)
                outputs = append_to_outputs(batch_outputs, outputs)
                callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
                if self.stop_predicting:
                    break
        callbacks.on_predict_end()
        outputs = tree.map_structure_up_to(
            batch_outputs, potentially_ragged_concat, outputs
        )
        return tf.nest.map_structure(convert_to_np_if_not_ragged, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        self.make_train_function()
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        def data():
            yield (x, y, sample_weight)

        logs = self.train_function(data())
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")
        self.make_test_function()

        def data():
            yield (x, y, sample_weight)

        logs = self.test_function(data())
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tf.nest.map_structure(
            convert_to_np_if_not_ragged, batch_outputs
        )
        return batch_outputs

    # Backwards compatibility shims.
    @property
    def compiled_metrics(self):
        class DeprecatedCompiledMetric:
            def update_state(_, y, y_pred, sample_weight=None):
                return self._compiled_metrics_update_state(
                    y, y_pred, sample_weight=sample_weight
                )

        return DeprecatedCompiledMetric()

    def _compiled_metrics_update_state(self, y, y_pred, sample_weight=None):
        warnings.warn(
            "`model.compiled_metrics()` is deprecated. "
            "Instead, use e.g.:\n"
            "```\n"
            "for metric in self.metrics:\n"
            "    metric.update_state(y, y_pred)\n"
            "```\n",
            stacklevel=2,
        )
        for metric in self.metrics:
            if isinstance(metric, metrics_module.Mean):
                metric.update_state(y_pred, sample_weight=sample_weight)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

    def compiled_loss(
        self, y, y_pred, sample_weight=None, regularization_losses=None
    ):
        warnings.warn(
            "`model.compiled_loss()` is deprecated. "
            "Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.",
        )
        return self.compute_loss(
            x=None, y=y, y_pred=y_pred, sample_weight=sample_weight
        )

    def loss(self, y, y_pred, sample_weight=None):
        warnings.warn(
            "`model.loss` is deprecated. "
            "Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.",
        )
        return self.compute_loss(
            x=None, y=y, y_pred=y_pred, sample_weight=sample_weight
        )


class TFEpochIterator(EpochIterator):
    def __init__(self, distribute_strategy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribute_strategy = distribute_strategy
        dataset = self._get_iterator()
        if not isinstance(dataset, tf.distribute.DistributedDataset):
            dataset = self._distribute_strategy.experimental_distribute_dataset(
                dataset
            )
        self._distributed_dataset = dataset
        self._steps_seen = 0

    def _get_iterator(self):
        return self.data_adapter.get_tf_dataset()

    def enumerate_epoch(self):
        if self.steps_per_epoch:
            if not self._current_iterator:
                self._current_iterator = iter(self._distributed_dataset)
            for step in range(
                0, self.steps_per_epoch, self.steps_per_execution
            ):
                yield step, self._current_iterator
        else:
            iterator = iter(self._distributed_dataset)
            if self.num_batches:
                for step in range(
                    0, self.num_batches, self.steps_per_execution
                ):
                    yield step, iterator
            else:
                step = -1
                while True:
                    step += self.steps_per_execution
                    self._steps_seen = step + 1
                    yield step, iterator
        self.data_adapter.on_epoch_end()

    def tf_sync(self):
        tf_context.async_wait()

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
            self.tf_sync()
        except (StopIteration, tf.errors.OutOfRangeError):
            if self._num_batches is None:
                self._num_batches = self._steps_seen
            warnings.warn(
                "Your input ran out of data; interrupting training. "
                "Make sure that your dataset or generator can generate "
                "at least `steps_per_epoch * epochs` batches. "
                "You may need to use the `.repeat()` "
                "function when building your dataset.",
                stacklevel=2,
            )
            self._current_iterator = None
            self.data_adapter.on_epoch_end()
```
[Go back to the beginning of the Section](#keras-tensorflow-trainer-class)


## <a id="keras-model-class"></a>Keras Model class

[Go back to Contents](#contents)


Excerpt from `keras/src/models/model.py`

```python
@keras_export(["keras.Model", "keras.models.Model"])
class Model(Trainer, Layer):
    """A model grouping layers into an object with training/inference features.

    There are three ways to instantiate a `Model`:

    ## With the "Functional API"

    You start from `Input`,
    you chain layer calls to specify the model's forward pass,
    and finally you create your model from inputs and outputs:

    ```python
    inputs = keras.Input(shape=(37,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    ```

    Note: Only dicts, lists, and tuples of input tensors are supported. Nested
    inputs are not supported (e.g. lists of list or dicts of dict).

    A new Functional API model can also be created by using the
    intermediate tensors. This enables you to quickly extract sub-components
    of the model.

    Example:

    ```python
    inputs = keras.Input(shape=(None, None, 3))
    processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
    conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
    pooling = keras.layers.GlobalAveragePooling2D()(conv)
    feature = keras.layers.Dense(10)(pooling)

    full_model = keras.Model(inputs, feature)
    backbone = keras.Model(processed, conv)
    activations = keras.Model(conv, feature)
    ```

    Note that the `backbone` and `activations` models are not
    created with `keras.Input` objects, but with the tensors that originate
    from `keras.Input` objects. Under the hood, the layers and weights will
    be shared across these models, so that user can train the `full_model`, and
    use `backbone` or `activations` to do feature extraction.
    The inputs and outputs of the model can be nested structures of tensors as
    well, and the created models are standard Functional API models that support
    all the existing APIs.

    ## By subclassing the `Model` class

    In that case, you should define your
    layers in `__init__()` and you should implement the model's forward pass
    in `call()`.

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation="relu")
            self.dense2 = keras.layers.Dense(5, activation="softmax")

        def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)

    model = MyModel()
    ```

    If you subclass `Model`, you can optionally have
    a `training` argument (boolean) in `call()`, which you can use to specify
    a different behavior in training and inference:

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation="relu")
            self.dense2 = keras.layers.Dense(5, activation="softmax")
            self.dropout = keras.layers.Dropout(0.5)

        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.dropout(x, training=training)
            return self.dense2(x)

    model = MyModel()
    ```

    Once the model is created, you can config the model with losses and metrics
    with `model.compile()`, train the model with `model.fit()`, or use the model
    to do prediction with `model.predict()`.

    ## With the `Sequential` class

    In addition, `keras.Sequential` is a special case of model where
    the model is purely a stack of single-input, single-output layers.

    ```python
    model = keras.Sequential([
        keras.Input(shape=(None, None, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=3),
    ])
    ```
    """

    def __new__(cls, *args, **kwargs):
        # Signature detection for usage of `Model` as a `Functional`
        if functional_init_arguments(args, kwargs) and cls == Model:
            from keras.src.models import functional

            return functional.Functional(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        from keras.src.models import functional

        # Signature detection for usage of a `Model` subclass
        # as a `Functional` subclass
        if functional_init_arguments(args, kwargs):
            inject_functional_model_class(self.__class__)
            functional.Functional.__init__(self, *args, **kwargs)
        else:
            Layer.__init__(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    @property
    def layers(self):
        return list(self._flatten_layers(include_self=False, recursive=False))

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Model.layers` attribute is reserved and should not be used. "
            "Please use another name."
        )

    @traceback_utils.filter_traceback
    def get_layer(self, name=None, index=None):
        """Retrieves a layer based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.
        Indices are based on order of horizontal graph traversal (bottom-up).

        Args:
            name: String, name of layer.
            index: Integer, index of layer.

        Returns:
            A layer instance.
        """
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a layer name or a layer index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError(
                    f"Was asked to retrieve layer at index {index}"
                    f" but model only has {len(self.layers)}"
                    " layers."
                )
            else:
                return self.layers[index]

        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(
                f"No such layer: {name}. Existing layers are: "
                f"{list(layer.name for layer in self.layers)}."
            )
        raise ValueError(
            "Provide either a layer name or layer index at `get_layer`."
        )

    @traceback_utils.filter_traceback
    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
        """Prints a string summary of the network.

        Args:
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided, becomes
                `[0.3, 0.6, 0.70, 1.]`. Defaults to `None`.
            print_fn: Print function to use. By default, prints to `stdout`.
                If `stdout` doesn't work in your environment, change to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            expand_nested: Whether to expand the nested models.
                Defaults to `False`.
            show_trainable: Whether to show if a layer is trainable.
                Defaults to `False`.
            layer_range: a list or tuple of 2 strings,
                which is the starting layer name and ending layer name
                (both inclusive) indicating the range of layers to be printed
                in summary. It also accepts regex patterns instead of exact
                name. In such case, start predicate will be the first element
                it matches to `layer_range[0]` and the end predicate will be
                the last element it matches to `layer_range[1]`.
                By default `None` which considers all layers of model.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        summary_utils.print_summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            layer_range=layer_range,
        )

    @traceback_utils.filter_traceback
    def save(self, filepath, overwrite=True, **kwargs):
        """Saves a model as a `.keras` file.

        Args:
            filepath: `str` or `pathlib.Path` object. Path where to save
                the model. Must end in `.keras`.
            overwrite: Whether we should overwrite any existing model at
                the target location, or instead ask the user via
                an interactive prompt.
            save_format: The `save_format` argument is deprecated in Keras 3.
                Format to use, as a string. Only the `"keras"` format is
                supported at this time.

        Example:

        ```python
        model = keras.Sequential(
            [
                keras.layers.Dense(5, input_shape=(3,)),
                keras.layers.Softmax(),
            ],
        )
        model.save("model.keras")
        loaded_model = keras.saving.load_model("model.keras")
        x = keras.random.uniform((10, 3))
        assert np.allclose(model.predict(x), loaded_model.predict(x))
        ```

        Note that `model.save()` is an alias for `keras.saving.save_model()`.

        The saved `.keras` file contains:

        - The model's configuration (architecture)
        - The model's weights
        - The model's optimizer's state (if any)

        Thus models can be reinstantiated in the exact same state.
        """
        return saving_api.save_model(self, filepath, overwrite, **kwargs)

    @traceback_utils.filter_traceback
    def save_weights(self, filepath, overwrite=True):
        """Saves all layer weights to a `.weights.h5` file.

        Args:
            filepath: `str` or `pathlib.Path` object.
                Path where to save the model. Must end in `.weights.h5`.
            overwrite: Whether we should overwrite any existing model
                at the target location, or instead ask the user
                via an interactive prompt.
        """
        if not str(filepath).endswith(".weights.h5"):
            raise ValueError(
                "The filename must end in `.weights.h5`. "
                f"Received: filepath={filepath}"
            )
        try:
            exists = os.path.exists(filepath)
        except TypeError:
            exists = False
        if exists and not overwrite:
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        saving_lib.save_weights_only(self, filepath)

    @traceback_utils.filter_traceback
    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        """Load weights from a file saved via `save_weights()`.

        Weights are loaded based on the network's
        topology. This means the architecture should be the same as when the
        weights were saved. Note that layers that don't have weights are not
        taken into account in the topological ordering, so adding or removing
        layers is fine as long as they don't have weights.

        **Partial weight loading**

        If you have modified your model, for instance by adding a new layer
        (with weights) or by changing the shape of the weights of a layer,
        you can choose to ignore errors and continue loading
        by setting `skip_mismatch=True`. In this case any layer with
        mismatching weights will be skipped. A warning will be displayed
        for each skipped layer.

        Args:
            filepath: String, path to the weights file to load.
                It can either be a `.weights.h5` file
                or a legacy `.h5` weights file.
            skip_mismatch: Boolean, whether to skip loading of layers where
                there is a mismatch in the number of weights, or a mismatch in
                the shape of the weights.
        """
        saving_api.load_weights(
            self, filepath, skip_mismatch=skip_mismatch, **kwargs
        )

    def build_from_config(self, config):
        if not config:
            return
        if "input_shape" in config:
            # Case: all inputs are in the first arg (possibly nested).
            if utils.is_default(self.build):
                status = self._build_by_run_for_single_pos_arg(
                    config["input_shape"]
                )
            else:
                try:
                    self.build(config["input_shape"])
                    status = True
                except:
                    status = False
            self._build_shapes_dict = config

        elif "shapes_dict" in config:
            # Case: inputs were recorded as multiple keyword arguments.
            if utils.is_default(self.build):
                status = self._build_by_run_for_kwargs(config["shapes_dict"])
            else:
                try:
                    self.build(**config["shapes_dict"])
                    status = True
                except:
                    status = False
            self._build_shapes_dict = config["shapes_dict"]

        if not status:
            warnings.warn(
                f"Model '{self.name}' had a build config, but the model "
                "cannot be built automatically in "
                "`build_from_config(config)`. "
                "You should implement "
                "`def build_from_config(self, config)`, "
                "and you might also want to implement the method "
                " that generates the config at saving time, "
                "`def get_build_config(self)`. "
                "The method `build_from_config()` is meant to "
                "create the state of the model (i.e. its variables) "
                "upon deserialization.",
                stacklevel=2,
            )

    def to_json(self, **kwargs):
        """Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={...})`.

        Args:
            **kwargs: Additional keyword arguments to be passed to
                `json.dumps()`.

        Returns:
            A JSON string.
        """
        from keras.src.saving import serialization_lib

        model_config = serialization_lib.serialize_keras_object(self)
        return json.dumps(model_config, **kwargs)

    def export(self, filepath, format="tf_saved_model"):
        """[TF backend only]* Create a TF SavedModel artifact for inference
        (e.g. via TF-Serving).

        **Note:** This can currently only be used with the TF backend.

        This method lets you export a model to a lightweight SavedModel artifact
        that contains the model's forward pass only (its `call()` method)
        and can be served via e.g. TF-Serving. The forward pass is registered
        under the name `serve()` (see example below).

        The original code of the model (including any custom layers you may
        have used) is *no longer* necessary to reload the artifact -- it is
        entirely standalone.

        Args:
            filepath: `str` or `pathlib.Path` object. Path where to save
                the artifact.

        Example:

        ```python
        # Create the artifact
        model.export("path/to/location")

        # Later, in a different process / environment...
        reloaded_artifact = tf.saved_model.load("path/to/location")
        predictions = reloaded_artifact.serve(input_data)
        ```

        If you would like to customize your serving endpoints, you can
        use the lower-level `keras.export.ExportArchive` class. The
        `export()` method relies on `ExportArchive` internally.
        """
        from keras.src.export import export_lib

        export_lib.export_model(self, filepath)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.src.models.functional import Functional

        functional_config_keys = [
            "name",
            "layers",
            "input_layers",
            "output_layers",
        ]
        is_functional_config = all(
            key in config for key in functional_config_keys
        )
        argspec = inspect.getfullargspec(cls.__init__)
        functional_init_args = inspect.getfullargspec(Functional.__init__).args[
            1:
        ]
        revivable_as_functional = (
            cls in {Functional, Model}
            or argspec.args[1:] == functional_init_args
            or (argspec.varargs == "args" and argspec.varkw == "kwargs")
        )
        if is_functional_config and revivable_as_functional:
            # Revive Functional model
            # (but not Functional subclasses with a custom __init__)
            from keras.src.models.functional import functional_from_config

            return functional_from_config(
                cls, config, custom_objects=custom_objects
            )

        # Either the model has a custom __init__, or the config
        # does not contain all the information necessary to
        # revive a Functional model. This happens when the user creates
        # subclassed models where `get_config()` is returning
        # insufficient information to be considered a Functional model.
        # In this case, we fall back to provide all config into the
        # constructor of the class.
        try:
            return cls(**config)
        except TypeError as e:
            raise TypeError(
                "Unable to revive model from config. When overriding "
                "the `get_config()` method, make sure that the "
                "returned config contains all items used as arguments "
                f"in the  constructor to {cls}, "
                "which is the default behavior. "
                "You can override this default behavior by defining a "
                "`from_config(cls, config)` class method to specify "
                "how to create an "
                f"instance of {cls.__name__} from its config.\n\n"
                f"Received config={config}\n\n"
                f"Error encountered during deserialization: {e}"
            )

    def _get_variable_map(self):
        store = {}
        map_trackable_variables(self, store=store, visited_trackables=set())
        return store


@keras_export("keras.models.model_from_json")
def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.

    Usage:

    >>> model = keras.Sequential([
    ...     keras.layers.Dense(5, input_shape=(3,)),
    ...     keras.layers.Softmax()])
    >>> config = model.to_json()
    >>> loaded_model = keras.models.model_from_json(config)

    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A Keras model instance (uncompiled).
    """
    from keras.src.saving import serialization_lib

    model_config = json.loads(json_string)
    return serialization_lib.deserialize_keras_object(
        model_config, custom_objects=custom_objects
    )


def functional_init_arguments(args, kwargs):
    return (
        (len(args) == 2)
        or (len(args) == 1 and "outputs" in kwargs)
        or ("inputs" in kwargs and "outputs" in kwargs)
    )


def inject_functional_model_class(cls):
    """Inject `Functional` into the hierarchy of this class if needed."""
    from keras.src.models import functional

    if cls == Model:
        return functional.Functional
    # In case there is any multiple inheritance, we stop injecting the
    # class if keras model is not in its class hierarchy.
    if cls == object:
        return object

    cls.__bases__ = tuple(
        inject_functional_model_class(base) for base in cls.__bases__
    )
    # Trigger any `__new__` class swapping that needed to happen on `Functional`
    # but did not because functional was not in the class hierarchy.
    cls.__new__(cls)

    return cls


Model.fit.__doc__ = base_trainer.Trainer.fit.__doc__
Model.predict.__doc__ = base_trainer.Trainer.predict.__doc__
Model.evaluate.__doc__ = base_trainer.Trainer.evaluate.__doc__
Model.train_on_batch.__doc__ = base_trainer.Trainer.train_on_batch.__doc__
Model.test_on_batch.__doc__ = base_trainer.Trainer.test_on_batch.__doc__
Model.predict_on_batch.__doc__ = base_trainer.Trainer.predict_on_batch.__doc__
```
[Go back to the beginning of the Section](#keras-model-class)
