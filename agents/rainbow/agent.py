#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from dqn_algo import DQN
from rainbow_dqn_model import rainbow_models

from sonic_util import AllowBacktracking, make_envs

def main():
    """Run DQN until the environment throws an exception."""
    envs = make_envs(stack=False, scale_rew=False)
    for i in range(len(envs)):
        envs[i] = AllowBacktracking(envs[i])
        envs[i] = BatchedFrameStack(BatchedGymEnv([[envs[i]]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        online_model, target_model = rainbow_models(sess,
                                  envs[0].action_space.n,
                                  gym_space_vectorizer(envs[0].observation_space),
                                  min_val=-200,
                                  max_val=200)
        replay_buffer = PrioritizedReplayBuffer(400000, 0.5, 0.4, epsilon=0.1)
        saver = tf.train.Saver()
        # either
        saver.restore(sess, '/root/compo/model')
        # or
        # sess.run(tf.global_variables_initializer())
        # end either
        dqn = DQN(online_model, target_model)
        players = []
        for env in envs:
            player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
            players.append(player)
        optimize = dqn.optimize(learning_rate=1e-4)
        while True:
            dqn.train(num_steps=16384,
                  players=players,
                  replay_buffer=replay_buffer,
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)
            saver.save(sess, '/root/compo/out/model')

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
