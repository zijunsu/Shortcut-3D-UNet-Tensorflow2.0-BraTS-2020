"""
Gradient checkpointing, adapted from the tf-slim implementation:
https://github.com/google-research/tf-slim/blob/a62dc893de5e46e6f2e9ec24a74b2abce026307a/tf_slim/layers/rev_block_lib.py

Example usage as decorator:
 
def call(self, x, past):
    @gradient_checkpointing.recompute_grad
    def inner(x):
        # ops go here
        return y
    return inner(x)
"""

import functools

import numpy as np
import six
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.ops import (array_ops, control_flow_ops, control_flow_util, custom_gradient, gradients_impl,
                                   math_ops, variable_scope)
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.util import tf_inspect

import arg_scope as arg_scope_lib

_WRONG_VARS_ERR = """\
# The variables used on recompute were different than the variables originally
# used. The function wrapped with @recompute_grad likely creates its own variable
# scope with a default name and has been called twice in the same enclosing scope.
# To fix, ensure each call to the function happens in its own unique variable scope.
# """


def _gradients(*args, **kwargs):
    return (gradients_impl.gradients_v2(*args, **kwargs) if tf2.enabled() else gradients_impl.gradients(
        *args, **kwargs))


def _safe_wraps(fn):
    if isinstance(fn, functools.partial):
        # functools.partial objects cannot be wrapped as they are missing the properties (__name__, __module__, __doc__).
        def passthrough(f):
            return f

        return passthrough
    return functools.wraps(fn)


def _scope_wrap(fn, scope):

    @_safe_wraps(fn)
    def wrap(*args, **kwargs):
        with variable_scope.variable_scope(scope, use_resource=True):
            return fn(*args, **kwargs)

    return wrap


def enable_with_args(dec):
    """A decorator for decorators to enable their usage with or without args."""

    @_safe_wraps(dec)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            # Used as decorator without args
            fn = args[0]
            return dec(fn)
        else:
            return lambda fn: dec(fn, *args, **kwargs)

    return new_dec


@enable_with_args
def recompute_grad(fn, use_data_dep=True, tupleize_grads=False):
    """Decorator that recomputes the function on the backwards pass.

    To use this function, you must use `ResourceVariable`s (i.e. `variable_scope(name, use_resource=True),
    which are the default in Eager mode and when running on TPU.

    Warning: Because the function will be called again on the backwards pass, the user should be careful to
    not use ops in their function that mutate state or have randomness (for example, batch normalization or
    dropout). If the function does have such operations, it is recommended that the function take the
    `is_recomputing` keyword argument which will be `False` on the forward pass and `True` on the backwards pass
    so that it can disable state changes when `is_recomputing=True` (for example, not updating the moving averages
    in batch normalization).

    Args:
      fn: a function that takes Tensors (all as positional arguments) and returns a tuple of Tensors.
      Note that `fn` should not close over any other Tensors or Variables.
      use_data_dep: `bool`, if `True` will use a dummy data dependency to force the recompute to happen.
      If `False` will use a control dependency. By default will be `True` if in an XLA context and `False` otherwise.
      XLA ignores control dependencies and so this data dependency is necessary.
      tupleize_grads: `bool`, if `True` will use control dependencies to ensure that all gradients are produced
      before any are consumed by downstream ops. If `use_data_dep` is also `True`, will use a data dependency
      instead of a control dependency.

    Returns:
      A wrapped fn that is identical to fn when called, but its activations will be discarded and recomputed
      on the backwards pass (i.e. on a call to tf.gradients).

    Raises:
      ValueError: if `fn` closes over any Tensors or Variables.
    """
    # Check for closed-over Tensors/Variables
    if fn.__code__.co_freevars:
        closed_over_vars = dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))
        for var_name, value in six.iteritems(closed_over_vars):
            if isinstance(value, (framework_ops.Tensor, variables_lib.Variable)):
                raise ValueError("fn decorated with @recompute_grad closes over Tensor %s "
                                 "(local variable name: %s). The decorated fn must not close over "
                                 "Tensors or Variables because gradients will NOT be computed for "
                                 "them through fn. To ensure correct gradients, make the "
                                 "Tensor an input to fn." % (value.name, var_name))

    @_safe_wraps(fn)
    def wrapped(*args):
        return _recompute_grad(fn, args, use_data_dep=True, tupleize_grads=False)

    return wrapped


def _is_on_tpu():
    return control_flow_util.GraphOrParentsInXlaContext(framework_ops.get_default_graph())


def _recomputing_grad_fn(compute_fn, original_args, original_vars, output_grads, grad_fn_variables, use_data_dep,
                         tupleize_grads, arg_scope, var_scope, has_is_recompute_kwarg):
    """Grad fn for recompute_grad."""
    variables = grad_fn_variables or []

    # Identity ops around the inputs ensures correct gradient graph-walking.
    inputs = [array_ops.identity(x) for x in list(original_args)]

    # Recompute outputs
    # Use a control dependency to ensure that the recompute is not eliminated by
    # CSE and that it happens on the backwards pass.
    ctrl_dep_grads = [g for g in output_grads if g is not None]
    with framework_ops.control_dependencies(ctrl_dep_grads):
        if use_data_dep:
            inputs = _force_data_dependency(output_grads, inputs)
        # Re-enter scopes
        with arg_scope_lib.arg_scope(arg_scope):
            with variable_scope.variable_scope(var_scope, reuse=True):
                # Re-call the function and ensure that the touched variables are the same as in the first call.
                with backprop.GradientTape() as tape:
                    fn_kwargs = {}
                    if has_is_recompute_kwarg:
                        fn_kwargs["is_recomputing"] = True
                    outputs = compute_fn(*inputs, **fn_kwargs)
                recompute_vars = set(_as_ref(v) for v in tape.watched_variables())
                if original_vars != recompute_vars:
                    raise ValueError(_WRONG_VARS_ERR)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs = list(outputs)

    # Compute gradients
    grads = _gradients(outputs, inputs + variables, output_grads, stop_gradients=inputs)

    if tupleize_grads:
        if use_data_dep:
            grads = _tuple_with_data_dep(grads)
        else:
            grads = control_flow_ops.tuple(grads)

    grad_inputs = grads[:len(inputs)]
    grad_vars = grads[len(inputs):]
    return grad_inputs, grad_vars


def _as_ref(v):
    # Tensorflow 2.1 no longer allows hashable variables. However tensorflow 2.0 still doesn't have experimental_ref.
    # This try to capture both cases.
    try:
        return v.experimental_ref()
    except AttributeError:
        return v


def _recompute_grad(fn, args, use_data_dep=True, tupleize_grads=False):
    """See recompute_grad."""
    has_is_recompute_kwarg = "is_recomputing" in tf_inspect.getargspec(fn).args
    for arg in args:
        if not isinstance(arg, (framework_ops.Tensor, variables_lib.Variable)):
            raise ValueError("All inputs to function must be Tensors")
    use_data_dep_ = use_data_dep
    if use_data_dep_:  # USE_DEFAULT == True
        use_data_dep_ = _is_on_tpu()

    # Use custom_gradient and return a grad_fn that recomputes on the backwards pass.
    @custom_gradient.custom_gradient
    def fn_with_recompute(*args):
        """Wrapper for fn."""
        # Capture the variable and arg scopes so we can re-enter them when recomputing.
        vs = variable_scope.get_variable_scope()
        arg_scope = arg_scope_lib.current_arg_scope()
        # Track all variables touched in the function.
        with backprop.GradientTape() as tape:
            fn_kwargs = {}
            if has_is_recompute_kwarg:
                fn_kwargs["is_recomputing"] = False
            outputs = fn(*args, **fn_kwargs)
        original_vars = set(_as_ref(v) for v in tape.watched_variables())

        def _grad_fn(output_grads, variables=None):
            # Validate that custom_gradient passes the right variables into grad_fn.
            if original_vars:
                assert variables, ("Fn created variables but the variables were not passed to the gradient fn.")
                if set(_as_ref(v) for v in variables) != original_vars:
                    raise ValueError(_WRONG_VARS_ERR)

            return _recomputing_grad_fn(compute_fn=fn,
                                        original_args=args,
                                        original_vars=original_vars,
                                        output_grads=output_grads,
                                        grad_fn_variables=variables,
                                        use_data_dep=use_data_dep_,
                                        tupleize_grads=tupleize_grads,
                                        arg_scope=arg_scope,
                                        var_scope=vs,
                                        has_is_recompute_kwarg=has_is_recompute_kwarg)

        # custom_gradient inspects the signature of the function to determine
        # whether the user expects variables passed in the grad_fn. If the function
        # created variables, the grad_fn should accept the "variables" kwarg.
        if original_vars:

            def grad_fn(*output_grads, **kwargs):
                return _grad_fn(output_grads, kwargs["variables"])
        else:

            def grad_fn(*output_grads):
                return _grad_fn(output_grads)

        return outputs, grad_fn

    return fn_with_recompute(*args)


def _force_data_dependency(first_compute, then_compute):
    """Force all of `then_compute` to depend on all of `first_compute`.

    Uses a dummy data dependency, which is useful when running on TPUs because
    XLA ignores control dependencies. Only supports float arguments.

    Args:
      first_compute: `list<Tensor>`. These will be made to run before the
        `Tensor`s `then_compute`.
      then_compute: `list<Tensor>`. These will run after all the `Tensor`s in
        `first_compute`.

    Returns:
      `list<Tensor>`, same length as `then_compute`.

    Raises:
      ValueError: if ranks are unknown or types are not floating.
    """

    def _first_element(x):
        if x.get_shape().ndims is None:
            raise ValueError("Rank of Tensor %s must be known" % x)
        ndims = x.get_shape().ndims
        begin = framework_ops.convert_to_tensor([0] * ndims, dtype=dtypes.int32)
        size = framework_ops.convert_to_tensor([1] * ndims, dtype=dtypes.int32)
        return array_ops.reshape(array_ops.slice(x, begin, size), [])

    first_compute_sum = math_ops.add_n([_first_element(x) for x in first_compute if x is not None])
    dtype = first_compute_sum.dtype
    if not dtype.is_floating:
        raise ValueError("_force_data_dependency only supports floating dtypes.")
    epsilon = np.finfo(dtype.as_numpy_dtype).tiny
    zero = array_ops.stop_gradient(epsilon * first_compute_sum)

    return [array_ops.identity(x) + zero if x is not None else None for x in then_compute]


def _tuple_with_data_dep(tensors):
    return _force_data_dependency(tensors, tensors)

