"""
Distributional Q-learning models.
"""

from functools import partial
from math import log

from math import sqrt

import numpy as np
import tensorflow as tf

# pylint: disable=R0913,R0903


def product(vals):
    """
    Compute the product of values in a list-like object.
    """
    prod = 1
    for val in vals:
        prod *= val
    return prod


def take_vector_elems(vectors, indices):
    """
    For a batch of vectors, take a single vector component
    out of each vector.

    Args:
      vectors: a [batch x dims] Tensor.
      indices: an int32 Tensor with `batch` entries.

    Returns:
      A Tensor with `batch` entries, one for each vector.
    """
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))


def nature_cnn(obs_batch, dense=tf.layers.dense):
    """
    Apply the CNN architecture from the Nature DQN paper.

    The result is a batch of feature vectors.
    """
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.orthogonal_initializer(gain=sqrt(2))
    }
    with tf.variable_scope('layer_1'):
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 8, 4, **conv_kwargs)
        print('float32 cnn_1 %s' % str(tf.shape(cnn_1)))
    with tf.variable_scope('layer_2'):
        cnn_2 = tf.layers.conv2d(cnn_1, 64, 4, 2, **conv_kwargs)
        print('float32 cnn_2 %s' % str(tf.shape(cnn_2)))
    with tf.variable_scope('layer_3'):
        cnn_3 = tf.layers.conv2d(cnn_2, 64, 3, 1, **conv_kwargs)
        print('float32 cnn_3 %s' % str(tf.shape(cnn_3)))
    flat_size = product([x.value for x in cnn_3.get_shape()[1:]])
    flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))
    print('float32 flat_in %s' % str(tf.shape(flat_in)))
    return dense(flat_in, 512, **conv_kwargs)


def noisy_net_dense(inputs,
                    units,
                    activation=None,
                    sigma0=0.5,
                    kernel_initializer=None,
                    name=None,
                    reuse=None):
    """
    Apply a factorized Noisy Net layer.

    See https://arxiv.org/abs/1706.10295.

    Args:
      inputs: the batch of input vectors.
      units: the number of output units.
      activation: the activation function.
      sigma0: initial stddev for the weight noise.
      kernel_initializer: initializer for kernels. Default
        is to use Gaussian noise that preserves stddev.
      name: the name for the layer.
      reuse: reuse the variable scope.
    """
    num_inputs = inputs.get_shape()[-1].value
    stddev = 1 / sqrt(num_inputs)
    if activation is None:
        activation = lambda x: x
    if kernel_initializer is None:
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    with tf.variable_scope(None, default_name=(name or 'noisy_layer'), reuse=reuse):
        weight_mean = tf.get_variable('weight_mu',
                                      shape=(num_inputs, units),
                                      initializer=kernel_initializer)
        bias_mean = tf.get_variable('bias_mu',
                                    shape=(units,),
                                    initializer=tf.zeros_initializer())
        stddev *= sigma0
        weight_stddev = tf.get_variable('weight_sigma',
                                        shape=(num_inputs, units),
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_stddev = tf.get_variable('bias_sigma',
                                      shape=(units,),
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_noise = tf.random_normal((units,), dtype=bias_stddev.dtype.base_dtype)
        weight_noise = _factorized_noise(num_inputs, units)
        return activation(tf.matmul(inputs, weight_mean + weight_stddev * weight_noise) +
                          bias_mean + bias_stddev * bias_noise)


def _factorized_noise(inputs, outputs):
    noise1 = _signed_sqrt(tf.random_normal((inputs, 1)))
    noise2 = _signed_sqrt(tf.random_normal((1, outputs)))
    return tf.matmul(noise1, noise2)


def _signed_sqrt(values):
    return tf.sqrt(tf.abs(values)) * tf.sign(values)


def rainbow_models(session,
                   num_actions,
                   obs_vectorizer,
                   num_atoms=51,
                   min_val=-10,
                   max_val=10,
                   sigma0=0.5):
    """
    Create the models used for Rainbow
    (https://arxiv.org/abs/1710.02298).

    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      num_atoms: number of distribution atoms.
      min_val: minimum atom value.
      max_val: maximum atom value.
      sigma0: initial Noisy Net noise.

    Returns:
      A tuple (online, target).
    """
    maker = lambda name: NatureDistQNetwork(session, num_actions, obs_vectorizer, name,
                                            num_atoms, min_val, max_val, dueling=True,
                                            dense=partial(noisy_net_dense, sigma0=sigma0))
    return maker('online'), maker('target')


class NatureDistQNetwork:
    """
    A distributional Q-network policy/value model based on the Nature
    DQN paper. Predicts action-conditional
    reward distributions (as opposed to expectations). Differentiable via
    TensorFlow.

    Attributes:
      session: the TF session for the model.
      num_actions: the size of the discrete action space.
      obs_vectorizer: used to convert observations into
        Tensors.
      name: the variable scope name for the network.
      variables: the trainable variables of the network.

    When a Q-network is instantiated, a graph is created
    that can be used by the step() method. This involves
    creating a set of variables and placeholders in the
    graph.

    After construction, other Q-network methods like
    transition_loss() reuse the variables that were made
    at construction time.
    """
    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 num_atoms,
                 min_val,
                 max_val,
                 dueling=False,
                 dense=tf.layers.dense,
                 input_dtype=tf.uint8,
                 input_scale=1 / 0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        self.session = session
        self.num_actions = num_actions
        self.obs_vectorizer = obs_vectorizer
        self.name = name
        self.dueling = dueling
        self.dense = dense
        self.dist = ActionDist(num_atoms, min_val, max_val)
        old_vars = tf.trainable_variables()
        with tf.variable_scope(name):
            self.step_obs_ph = tf.placeholder(self.input_dtype,
                                              shape=(None,) + obs_vectorizer.out_shape)
            print('uint8 tf.shape(self.step_obs_ph) = %s' % str(tf.shape(self.step_obs_ph)))
            self.step_base_out = self.base(self.step_obs_ph)
            log_probs = self.value_func(self.step_base_out)
            values = self.dist.mean(log_probs)
            self.step_outs = (values, log_probs)
        self.variables = [v for v in tf.trainable_variables() if v not in old_vars]

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        feed = self.step_feed_dict(observations, states)
        values, dists = self.session.run(self.step_outs, feed_dict=feed)
        return {
            'actions': np.argmax(values, axis=1),
            'states': None,
            'action_values': values,
            'action_dists': dists
        }

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        with tf.variable_scope(self.name, reuse=True):
            max_actions = tf.argmax(self.dist.mean(self.value_func(self.base(new_obses))),
                                    axis=1, output_type=tf.int32)
        with tf.variable_scope(target_net.name, reuse=True):
            target_preds = target_net.value_func(target_net.base(new_obses))
            target_preds = tf.where(terminals,
                                    tf.zeros_like(target_preds) - log(self.dist.num_atoms),
                                    target_preds)
        discounts = tf.where(terminals, tf.zeros_like(discounts), discounts)
        target_dists = self.dist.add_rewards(tf.exp(take_vector_elems(target_preds, max_actions)),
                                             rews, discounts)
        with tf.variable_scope(self.name, reuse=True):
            online_preds = self.value_func(self.base(obses))
            onlines = take_vector_elems(online_preds, actions)
            return _kl_divergence(tf.stop_gradient(target_dists), onlines)

    def value_func(self, feature_batch):
        """
        Go from a 2-D Tensor of feature vectors to a 3-D
        Tensor of predicted action distributions.

        Args:
          feature_batch: a batch of features from base().

        Returns:
          A Tensor of shape [batch x actions x atoms].

        All probabilities are computed in the log domain.
        """
        logits = self.dense(feature_batch, self.num_actions * self.dist.num_atoms)
        actions = tf.reshape(logits, (tf.shape(logits)[0], self.num_actions, self.dist.num_atoms))
        if not self.dueling:
            return tf.nn.log_softmax(actions)
        values = tf.expand_dims(self.dense(feature_batch, self.dist.num_atoms), axis=1)
        actions -= tf.reduce_mean(actions, axis=1, keepdims=True)
        return tf.nn.log_softmax(values + actions)

    # pylint: disable=W0613
    def step_feed_dict(self, observations, states):
        """Produce a feed_dict for taking a step."""
        return {self.step_obs_ph: self.obs_vectorizer.to_vecs(observations)}

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        """
        Go from a Tensor of observations to a Tensor of
        feature vectors to feed into the output heads.

        Returns:
          A Tensor of shape [batch_size x num_features].
        """
        obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        print('float32 tf.shape(obs_batch) = %s' % str(tf.shape(obs_batch)))
        return nature_cnn(obs_batch, dense=self.dense)

    def clear_top_weights(self):
        old_vars = tf.trainable_variables()
        with tf.variable_scope(self.name):
            log_probs = self.value_func(self.step_base_out)
            values = self.dist.mean(log_probs)
            self.step_outs = (values, log_probs)
        self.variables = [v for v in tf.trainable_variables() if v not in old_vars]


class ActionDist:
    """
    A discrete reward distribution.
    """
    def __init__(self, num_atoms, min_val, max_val):
        assert num_atoms >= 2
        assert max_val > min_val
        self.num_atoms = num_atoms
        self.min_val = min_val
        self.max_val = max_val
        self._delta = (self.max_val - self.min_val) / (self.num_atoms - 1)

    def atom_values(self):
        """Get the reward values for each atom."""
        return [self.min_val + i * self._delta for i in range(0, self.num_atoms)]

    def mean(self, log_probs):
        """Get the mean rewards for the distributions."""
        probs = tf.exp(log_probs)
        return tf.reduce_sum(probs * tf.constant(self.atom_values(), dtype=probs.dtype), axis=-1)

    def add_rewards(self, probs, rewards, discounts):
        """
        Compute new distributions after adding rewards to
        old distributions.

        Args:
          log_probs: a batch of log probability vectors.
          rewards: a batch of rewards.
          discounts: the discount factors to apply to the
            distribution rewards.

        Returns:
          A new batch of log probability vectors.
        """
        atom_rews = tf.tile(tf.constant([self.atom_values()], dtype=probs.dtype),
                            tf.stack([tf.shape(rewards)[0], 1]))

        fuzzy_idxs = tf.expand_dims(rewards, axis=1) + tf.expand_dims(discounts, axis=1) * atom_rews
        fuzzy_idxs = (fuzzy_idxs - self.min_val) / self._delta

        # If the position were exactly 0, rounding up
        # and subtracting 1 would cause problems.
        fuzzy_idxs = tf.clip_by_value(fuzzy_idxs, 1e-18, float(self.num_atoms - 1))

        indices_1 = tf.cast(tf.ceil(fuzzy_idxs) - 1, tf.int32)
        fracs_1 = tf.abs(tf.ceil(fuzzy_idxs) - fuzzy_idxs)
        indices_2 = indices_1 + 1
        fracs_2 = 1 - fracs_1

        res = tf.zeros_like(probs)
        for indices, fracs in [(indices_1, fracs_1), (indices_2, fracs_2)]:
            index_matrix = tf.expand_dims(tf.range(tf.shape(indices)[0], dtype=tf.int32), axis=1)
            index_matrix = tf.tile(index_matrix, (1, self.num_atoms))
            scatter_indices = tf.stack([index_matrix, indices], axis=-1)
            res = res + tf.scatter_nd(scatter_indices, probs * fracs, tf.shape(res))

        return res


def _kl_divergence(probs, log_probs):
    masked_diff = tf.where(tf.equal(probs, 0), tf.zeros_like(probs), tf.log(probs) - log_probs)
    return tf.reduce_sum(probs * masked_diff, axis=-1)
