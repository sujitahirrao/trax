# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Core layer types, such as `Dense`, `Embedding`, and `Dropout`."""

import functools
import inspect
from trax.layers import base
from trax.layers import combinators

# TODO(jaszczur): in decorator and AssertFunction add a check on number of
# inputs/outputs


def ash(spec):
  """TODO(jaszczur): function docstring."""
  caller = inspect.getframeinfo(inspect.stack()[1][0])
  message = f'Defined at {caller.filename}:{caller.lineno}'

  def wrap_cls(cls):
    forward = getattr(cls, 'forward')
    init = getattr(cls, '__init__')

    before_spec, after_spec = spec.split('->')

    @functools.wraps(init)
    def init_wrapper(self, *args, **kwargs):
      before_assert = AssertShape(before_spec,
                                  message=message + ' function input')
      after_assert = AssertShape(
          after_spec, linked_defined_shapes=before_assert.defined_shapes,
          message=message + ' function output')
      out = init(self, *args, **kwargs)
      self._before_assert_fun = before_assert  # pylint: disable=protected-access
      self._after_assert_fun = after_assert  # pylint: disable=protected-access
      return out

    @functools.wraps(forward)
    def forward_wrapper(self, x, *args, **kwargs):
      x = self._before_assert_fun.forward(x)  # pylint: disable=protected-access
      y = forward(self, x, *args, **kwargs)
      y = self._after_assert_fun.forward(y)  # pylint: disable=protected-access
      return y

    setattr(cls, 'forward', forward_wrapper)
    setattr(cls, '__init__', init_wrapper)
    return cls

  # TODO(jaszczur): replace this with forward/init override, always.
  def wrap_fun(fun):
    @functools.wraps(fun)
    def fun_wrapper(*args, **kwargs):
      layer = fun(*args, **kwargs)
      return AssertFunction(spec, layer, message)
    return fun_wrapper

  def wrap_fun_or_cls(fun_or_cls):
    if inspect.isclass(fun_or_cls):
      return wrap_cls(fun_or_cls)
    return wrap_fun(fun_or_cls)
  return wrap_fun_or_cls


def AssertFunction(spec, layer, message=None):  # pylint: disable=invalid-name
  """TODO(jaszczur): function docstring."""
  # TODO(jaszczur): How to handle no input or no output? by default an empty
  # string is interpreted as a 0d tensor (scalar).
  if message is None:
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    message = f'Defined at {caller.filename}:{caller.lineno}'
  before_spec, after_spec = spec.split('->')
  before_assert = AssertShape(before_spec, message=message + ' function input')
  after_assert = AssertShape(
      after_spec, linked_defined_shapes=before_assert.defined_shapes,
      message=message + ' function output')
  return combinators.Serial(
      before_assert, layer, after_assert)


class AssertShape(base.Layer):
  """Assert Shape layer.
  """

  def __init__(self, spec, linked_defined_shapes=None, message=None):
    """Something.

    Args:
      spec: eh, missing documentation is not good, is it?
      linked_defined_shapes: still missing
      message: still missing
    """
    super().__init__(name='AssertShape')
    spec = spec.replace('...', '*')
    for letter in spec:
      assert letter in '*,qwertasdfgzxcvbyuophjklnmQWERTASDFGZXCVBYUIOPHJKLNM1234567890'
    self._specs = spec.split(',')
    self._n_in = self._n_out = len(self._specs)
    if linked_defined_shapes is None:
      self.defined_shapes = {str(i): i for i in range(10)}
      self.linked = False
    else:
      self.defined_shapes = linked_defined_shapes
      self.linked = True

    if message is None:
      caller = inspect.getframeinfo(inspect.stack()[1][0])
      self.message = f'Defined at {caller.filename}:{caller.lineno}'
    else:
      self.message = message

  def forward(self, xs):
    if not self.linked:
      for x in list(self.defined_shapes.keys()):
        if not x.isdigit():
          del self.defined_shapes[x]

    if not isinstance(xs, (list, tuple)):
      xs = [xs]

    def assert_true(cond):
      if not cond:
        shapes = [x.shape for x in xs]
        # TODO(jaszczur): add line reporting.
        raise ValueError(f'AssertShape Error. Expected {self._specs}, got '
                         f'{shapes} with dict {self.defined_shapes}. '
                         f'{self.message}')

    def assert_equal(a, b):
      # It should, potentially, return the more specific dimension.
      assert_true(a == b)
      return a

    assert_equal(len(xs), len(self._specs))

    def check_shape(shape, spec):
      assert_equal(len(shape), len(spec))
      for shape_dim, letter in zip(shape, spec):
        if letter in self.defined_shapes:
          self.defined_shapes[letter] = assert_equal(
              self.defined_shapes[letter], shape_dim)
        else:
          self.defined_shapes[letter] = shape_dim

    def check_star(shape):
      if '*' not in self.defined_shapes:
        self.defined_shapes['*'] = shape
      else:
        assert_equal(len(shape), len(self.defined_shapes['*']))
        for s1, s2 in zip(shape, self.defined_shapes['*']):
          assert_equal(s1, s2)

    for x, spec in zip(xs, self._specs):
      if '*' in spec:
        assert_true(len(x.shape) >= (len(spec) - 1))
        before, after = spec.split('*')
        check_shape(x.shape[:len(before)], before)
        if after:
          check_shape(x.shape[-len(after):], after)
          check_star(x.shape[len(before):-len(after)])
        else:
          # if len(after) == 0 then -len(after) in indices evaluates badly.
          check_star(x.shape[len(before):])

      else:
        check_shape(x.shape, spec)

    if len(xs) == 1:
      return xs[0]
    else:
      return xs
