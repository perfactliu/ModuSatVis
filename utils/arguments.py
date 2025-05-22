import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-batch', type=int, default=50, help='the number of batches in one epoch')
    parser.add_argument('--network-learn-freq', type=int, default=40, help='the number to learn in one epoch')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')  # 256
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor of bellman')
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=3e-4, help='the learning rate of the critic')
    parser.add_argument('--change-goal-cycle', type=int, default=10, help='the number of cycles to change the goal')
    parser.add_argument('--max-cycle-steps', type=int, default=16, help='the max number of steps within one cycle')
    parser.add_argument('--layer-norm', type=bool, default=True, help='whether to use layer normalization')
    parser.add_argument('--tau', type=float, default=0.005, help='soft update rate of target networks')
    parser.add_argument('--target-update-freq', type=int, default=3, help='update frequency of target networks')
    parser.add_argument('--test_episodes', type=int, default=100, help='number of episodes to test')
    parser.add_argument('--plot', type=bool, default=False, help='whether to plot')

    args = parser.parse_args()

    return args
