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
"""Tests for core layers."""

from absl.testing import absltest
import numpy as np

import trax.layers as tl


class AssertFunctionTest(absltest.TestCase):
  """Test AssertFunction layer."""

  def test_pass1(self):
    layer = tl.AssertFunction('abc->abc', tl.Dropout(rate=0.1))
    x = np.ones((2, 5, 20))
    layer(x)

  def test_fail1(self):
    layer = tl.AssertFunction('abc->cba', tl.Dropout(rate=0.1))
    x = np.ones((2, 5, 20))
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_flat_pass1(self):
    layer = tl.AssertFunction('...ab->...c', tl.Flatten(n_axes_to_keep=3))
    x = np.ones((1, 2, 3, 4, 5))
    layer(x)

  def test_flat_pass2(self):
    layer = tl.AssertFunction('xyzab->xyzc', tl.Flatten(n_axes_to_keep=3))
    x = np.ones((1, 2, 3, 4, 5))
    layer(x)

  def test_flat_pass3(self):
    layer = tl.AssertFunction('abcde->x', tl.Flatten(n_axes_to_keep=0))
    x = np.ones((1, 2, 3, 4, 5))
    layer(x)

  def test_flat_pass4(self):
    layer = tl.AssertFunction('abcde->abcx', tl.Flatten(n_axes_to_keep=3))
    x = np.ones((1, 2, 3, 4, 5))
    layer(x)

  def test_flat_fail1(self):
    layer = tl.AssertFunction('abcde->abcde', tl.Flatten(n_axes_to_keep=3))
    x = np.ones((1, 2, 3, 4, 5))
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_flat_fail2(self):
    layer = tl.AssertFunction('abcde->abcd', tl.Flatten(n_axes_to_keep=3))
    x = np.ones((1, 2, 3, 4, 5))
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_branch_pass1(self):
    layer = tl.AssertFunction(
        '*cd->*x,*cd',
        tl.Branch(
            tl.Flatten(n_axes_to_keep=2),
            tl.Dropout(rate=0.1),
            ))
    x = np.ones((1, 2, 3, 4))
    layer(x)

  def test_branch_pass2(self):
    layer = tl.AssertFunction(
        '*34->1234,*34',
        tl.Branch(
            tl.Dropout(rate=0.1),
            tl.Serial(),
            ))
    x = np.ones((1, 2, 3, 4))
    layer(x)

  def test_branch_fail1(self):
    layer = tl.AssertFunction(
        '*cd->*x,*cd,*cd,*cd',
        tl.Branch(
            tl.Flatten(n_axes_to_keep=2),
            tl.Dropout(rate=0.1),
            tl.Serial(),
            ))
    x = np.ones((1, 2, 3, 4))
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_branch_fail2(self):
    layer = tl.AssertFunction(
        '*34->*x,*y',
        tl.Branch(
            tl.Flatten(n_axes_to_keep=3),
            tl.Serial(),
            ))
    x = np.ones((1, 2, 3, 4))
    with self.assertRaises(tl.LayerError):
      layer(x)


class AssertShapeTest(absltest.TestCase):
  """Test AssertShape layer."""

  def test_pass1(self):
    layer = tl.AssertShape('aba,ba')
    x = [np.ones((10, 5, 10)), np.zeros((5, 10))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_pass2(self):
    layer = tl.AssertShape('aba,ba')
    x = [np.ones((5, 5, 5)), np.zeros((5, 5))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_pass3(self):
    layer = tl.AssertShape('a')
    x = np.ones((5,))
    y = layer(x)
    self.assertEqual(y.tolist(), x.tolist())

  def test_pass3a(self):
    layer = tl.AssertShape('')
    x = np.ones(())
    y = layer(x)
    self.assertEqual(y.tolist(), x.tolist())

  def test_pass3b(self):
    layer = tl.AssertShape('aa')
    x = np.ones((3, 3))
    y = layer(x)
    self.assertEqual(y.tolist(), x.tolist())

  def test_pass4(self):
    layer = tl.AssertShape('a,')
    x = [np.ones((5,)), np.zeros(())]
    y = layer(x)
    self.assertEqual(y, x)

  def test_pass5(self):
    layer = tl.AssertShape('a,b,a')
    x = [np.ones((5,)), np.zeros((2)), np.zeros((5))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_pass6(self):
    layer = tl.AssertShape('a,b,a,ab')
    x = [np.ones((5,)), np.zeros((2)), np.zeros((5)), np.zeros((5, 2))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_num_pass1(self):
    layer = tl.AssertShape('23,1,93')
    x = [np.ones((2, 3)), np.zeros((1)), np.zeros((9, 3))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_num_fail1(self):
    layer = tl.AssertShape('24,1,93')
    x = [np.ones((2, 3)), np.zeros((1)), np.zeros((9, 3))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_star_pass1(self):
    layer = tl.AssertShape('a*bc,abc')
    x = [np.ones((1, 5, 5, 2, 3)), np.zeros((1, 2, 3))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_star_pass2(self):
    layer = tl.AssertShape('*bc,abc')
    x = [np.ones((5, 5, 2, 3)), np.zeros((1, 2, 3))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_star_pass3(self):
    layer = tl.AssertShape('*bc,abc')
    x = [np.ones((2, 3)), np.zeros((1, 2, 3))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_star_pass4(self):
    layer = tl.AssertShape('*bc,*bc')
    x = [np.ones((1, 2, 3)), np.zeros((1, 2, 3))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_star_pass5(self):
    layer = tl.AssertShape('bb*,*bb')
    x = [np.ones((2, 2, 5, 6)), np.zeros((5, 6, 2, 2))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_star_fail1(self):
    layer = tl.AssertShape('ab*cde,2')
    x = [np.ones((2, 3, 4, 5)), np.zeros((2))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_star_fail2(self):
    layer = tl.AssertShape('b*c,2')
    x = [np.ones((2)), np.zeros((2))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_star_fail3(self):
    layer = tl.AssertShape('b**c,2')
    x = [np.ones((2, 3, 4, 5)), np.zeros((2))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_star_fail4(self):
    layer = tl.AssertShape('*a,*b')
    x = [np.ones((1, 2, 3, 7)), np.zeros((1, 2, 8))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_ellipsis_pass1(self):
    layer = tl.AssertShape('...22,...3')
    x = [np.ones((1, 2, 3, 2, 2)), np.zeros((1, 2, 3, 3))]
    y = layer(x)
    self.assertEqual(y, x)

  def test_ellipsis_fail1(self):
    layer = tl.AssertShape('...c...,2')
    x = [np.ones((2, 3, 4, 5)), np.zeros((2))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_ellipsis_fail2(self):
    layer = tl.AssertShape('...abc,2')
    x = [np.ones((4, 5)), np.zeros((2))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_ellipsis_fail3(self):
    layer = tl.AssertShape('...2,...8')
    x = [np.ones((1, 2, 3, 9)), np.zeros((1, 3, 3, 8))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_fail(self):
    layer = tl.AssertShape('aba,ab')
    x = [np.ones((10, 5, 10)), np.ones((5, 8))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_fail2(self):
    layer = tl.AssertShape('aba,ab')
    x = [np.ones((10, 5, 10)), np.ones((5, 10, 4))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_fail3(self):
    layer = tl.AssertShape('aba,ab')
    x = [np.ones((10, 5, 10)), np.ones((10, 6,))]
    with self.assertRaises(tl.LayerError):
      layer(x)

  def test_fail4(self):
    layer = tl.AssertShape('aba,ab')
    x = [np.ones((10, 5, 5)), np.ones((10, 5,))]
    with self.assertRaises(tl.LayerError):
      layer(x)

if __name__ == '__main__':
  absltest.main()
