# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import random

import go

import numpy as np

"""
Allowable symmetries:
identity [12][34]
rot90 [24][13]
rot180 [43][21]
rot270 [31][42]
flip [13][24]
fliprot90 [34][12]
fliprot180 [42][31]
fliprot270 [21][43]
"""
INVERSES = {
    'identity': 'identity',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90',
    'flip': 'flip',
    'fliprot90': 'fliprot90',
    'fliprot180': 'fliprot180',
    'fliprot270': 'fliprot270',
}

IMPLS = {
    'identity': lambda x: x,
    'rot90': np.rot90,
    'rot180': functools.partial(np.rot90, k=2),
    'rot270': functools.partial(np.rot90, k=3),
    'flip': lambda x: np.rot90(np.fliplr(x)),
    'fliprot90': np.flipud,
    'fliprot180': lambda x: np.rot90(np.flipud(x)),
    'fliprot270': np.fliplr,
}

assert set(IMPLS.keys()) == set(INVERSES.keys())

# A symmetry is just a string describing the transformation.
SYMMETRIES = list(INVERSES.keys())


def invert_symmetry(s):
    return INVERSES[s]


def apply_symmetry_feat(sym, features):
    return IMPLS[sym](features)


def apply_symmetry_pi(s, pi):
    pi = np.copy(pi)
    # rotate all moves except for the pass move at end
    pi[:-1] = IMPLS[s](pi[:-1].reshape([go.N, go.N])).ravel()
    return pi


def randomize_symmetries_feat(features):
    symmetries_used = [random.choice(SYMMETRIES) for _ in features]
    return symmetries_used, [apply_symmetry_feat(s, f)
                             for s, f in zip(symmetries_used, features)]


def invert_symmetries_pi(symmetries, pis):
    return [apply_symmetry_pi(invert_symmetry(s), pi)
            for s, pi in zip(symmetries, pis)]


def rotate_train_nhwc(x, pi):
    sym = np.random.randint(low = 0, high = len(SYMMETRIES), size = ())

    def rotate(tensor):
        # flipLeftRight
        if np.bitwise_and(sym, 1) > 0:
          tensor = np.flip(tensor, axis = 0)
        # flipUpDown
        if np.bitwise_and(sym, 2) > 0:
          tensor = np.flip(tensor, axis = 1)
        # flipDiagonal
        if np.bitwise_and(sym, 4) > 0:
          tensor = np.transpose(tensor, axes = (1,0,2))
        return tensor

    # TODO(tommadams): use tf.ensure_shape instead of tf.assert_equal.
    squares = go.N * go.N
    assert np.all(pi.shape == np.array([squares + 1]))

    x_shape = x.shape.as_list()
    assert np.all(x.shape == np.array([go.N, go.N, x_shape[2]]))

    pi_move = pi[:squares]
    pi_pass = pi[squares:squares+1]
    # Add a final dim so that x and pi have same shape: [N,N,num_features].
    pi_n_by_n = np.reshape(pi_move, [go.N, go.N, 1])

    pi_rot = np.concatenate(
        [np.reshape(rotate(pi_n_by_n), [squares]), pi_pass],
        axis=0)

    return rotate(x), pi_rot


def rotate_train_nchw(x, pi):
    sym = np.random.randint(low = 0, high = len(SYMMETRIES), size = ())

    def rotate(tensor):
        # flipLeftRight
        if np.bitwise_and(sym, 1) > 0:
          tensor = np.flip(tensor, axis = 1)
        # flipUpDown
        if np.bitwise_and(sym, 2) > 0:
          tensor = np.flip(tensor, axis = 2)
        # flipDiagonal
        if np.bitwise_and(sym, 4) > 0:
          tensor = np.transpose(tensor, axes = (0,2,1))
        return tensor

    # TODO(tommadams): use tf.ensure_shape instead of tf.assert_equal.
    squares = go.N * go.N
    assert np.all(pi.shape == np.array([squares + 1]))

    x_shape = x.shape.as_list()
    assert np.all(x.shape == np.array([x_shape[0], go.N, go.N]))

    pi_move = pi[:squares]
    pi_pass = pi[squares:squares+1]
    # Add a dim so that x and pi have same shape: [num_features,N,N].
    pi_n_by_n = np.reshape(pi_move, [1, go.N, go.N])

    pi_rot = np.concatenate(
        [np.reshape(rotate(pi_n_by_n), [squares]), pi_pass],
        axis=0)

    return rotate(x), pi_rot
