from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0, 1, 2), (3, 4, 5), (6, 7, 8),  # horizontal
                         (0, 3, 6), (1, 4, 7), (2, 5, 8),  # vertical
                         (0, 4, 8), (2, 4, 6)])  # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9)  # grid: 1-D array of 9 elements
        self.turn = 1  # whose turn it is: 1:x; 2:o
        self.done = False  # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0: '.', 1: 'x', 2: 'o'}  # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """
        Mark a point on position action.
        :param action: the action to take (0 - 8)
        :type action: int
        :return (grid, status, done?)
        :rtype tuple
        """
        assert type(action) == int and 0 <= action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done


class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """

    def __init__(self, input_size = 27, hidden_size = 256, output_size = 9):
        super(Policy, self).__init__()
        # TODO
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        # TODO
        return self.network(x)


def select_action(policy, state):
    """Samples an action from the policy at the state.
    :param policy: policy
    :type policy: Policy
    :param state: the states (Environment.grid)
    :type state: ndarray
    :return (action (int), log_prob(action))
    :rtype tuple
    """
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3, 9).scatter_(0, state, 1).view(1, 27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob


def compute_returns(rewards, gamma = 1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    # TODO
    res = []
    for i in range(len(rewards)):
        curr_return = 0
        cur_rewards = rewards[i:]
        for j in range(len(cur_rewards)):
            curr_return += cur_rewards[j] * (gamma ** j)
        res.append(curr_return)
    return res


def finish_episode(saved_rewards, saved_logprobs, gamma = 1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph = True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step


def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
        Environment.STATUS_VALID_MOVE: 5,  # TODO
        Environment.STATUS_INVALID_MOVE: -50,
        Environment.STATUS_WIN: 100,
        Environment.STATUS_TIE: -10,
        Environment.STATUS_LOSE: -100
    }[status]


def train(policy, env, lr = 0.0005, gamma = 0.9, max_iter = 50000, verbose = True, plot = False, save_policy = True):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = 10000, gamma = 0.9)
    running_reward = 0

    # Parameters
    log_interval = max_iter / 100

    # Records
    iters = []
    avg_returns = []

    print("{} Start Training {}".format("=" * 20, "=" * 20))
    print("Parameters: lr = {}, gamma = {}, max_iter = {}".format(lr, gamma, max_iter))
    print("=" * 30)

    for i_episode in range(max_iter):
        saved_rewards = []
        saved_logprobs = []

        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            avg_return = running_reward / log_interval
            if verbose:
                results = play_games(env, policy)
                print('Episode {}\t'
                      'Average return: {:.2f}'
                      '\tInvalid: {:.4f}'.format(i_episode, avg_return, results[-1]))
            running_reward = 0
            iters.append(i_episode)
            avg_returns.append(avg_return)

            if save_policy:
                torch.save(policy.state_dict(), "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    print("{} Done Training {}".format("=" * 20, "=" * 20))
    result = play_games(env, policy, 1000)
    print("Result: {} / {} / {}".format(result[0], result[1], result[2]))

    if plot:
        plt.plot(iters, avg_returns, color = "b")
        plt.ylabel("Average Returns")
        plt.xlabel("Episodes")
        plt.title("Training Curve")
        plt.savefig("./Report/images/5/training_curve.png")
        # plt.show()
    return


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3, 9).scatter_(0, state, 1).view(1, 27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def play_games(env, policy, rounds = 100, visualize = False):
    """
    Get the policy results by playing rounds and count the results
    :param env: the environment
    :type env: Environment
    :param policy: the policy
    :type policy: Policy
    :param rounds: the number of rounds to play
    :type rounds: int
    :return: (num_policy_win, num_rand_win, num_tie)
    :rtype: tuple
    """
    num_policy_win, num_rand_win, num_tie = 0, 0, 0
    total_move, invalid_move = 0, 0

    for i in range(rounds):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            if status == Environment.STATUS_INVALID_MOVE:
                invalid_move += 1
            total_move += 1
            if visualize:
                env.render()

        if status == Environment.STATUS_WIN:
            num_policy_win += 1
        elif status == Environment.STATUS_LOSE:
            num_rand_win += 1
        elif status == Environment.STATUS_TIE:
            num_tie += 1
        else:
            print("Wrong status: {}".format(status))
            return

    return num_policy_win, num_rand_win, num_tie, invalid_move / float(total_move)


# ==================== Answers ====================
def part1(env):
    """
    Play a round of tictactoe using step and render.
    :param env: the environment
    :type env: Environment
    :return: None
    :rtype: None
    """
    assert all(i == 0 for i in env.grid)
    for i in range(9):
        env.step(i)
        env.render()
    return


def part2():
    return


def part3():
    import doctest
    doctest.testmod()
    return


def part4():
    return


def part5(env, policy, part_b = False):
    # a, c
    train(policy, env, gamma = 0.9, max_iter = 50000, plot = True)
    # b
    if part_b:
        best = []
        for hidden_dim in [32, 64, 128, 256]:
            test_env = Environment()
            test_policy = Policy(hidden_size = hidden_dim)
            train(test_policy, test_env, gamma = 0.9, max_iter = 50000, verbose = False,
                  plot = False, save_policy = False)
            results = play_games(test_env, test_policy, rounds = 1000)
            best.append(results)
        print([32, 64, 128, 256], best)
    # d
    results = play_games(env, policy, rounds = 100)
    print("win: {}, lose: {}, tie: {}", results[0], results[1], results[2])
    for i in range(5):
        print("{} Game {} {}".format("=" * 20, i, "=" * 20))
        play_games(env, policy, rounds = 1, visualize = True)
    return


def part6(env):
    iters, wins, loses, ties = [], [], [], []
    for ep in range(0, 50000, 1000):
        policy = Policy()
        load_weights(policy, ep)
        results = play_games(env, policy, rounds = 100, visualize = False)
        iters.append(ep)
        wins.append(results[0])
        loses.append(results[1])
        ties.append(results[2])
    plt.plot(iters, wins, color = "g", label = "win")
    plt.plot(iters, loses, color = "r", label = "lose")
    plt.plot(iters, ties, color = "b", label = "tie")
    plt.legend(loc = "best")
    plt.ylabel("Game Counts")
    plt.xlabel("Episodes")
    plt.title("Performance over episodes")
    plt.savefig("./Report/images/6/performance_curve.png")
    plt.show()
    return


def part7(env):
    iters= []
    moves = [[] for i in range(9)]
    for ep in range(0, 50000, 1000):
        policy = Policy()
        load_weights(policy, ep)
        move_distr = first_move_distr(policy, env)
        move_distr = np.array(move_distr)[0]
        for i in range(len(move_distr)):
            moves[i].append(move_distr[i])
        iters.append(ep)
    for i in range(len(moves)):
        plt.clf()
        plt.cla()
        plt.plot(iters, moves[i], label = "move_{}".format(i + 1))
        plt.ylabel("probability")
        plt.xlabel("Episodes")
        plt.legend(loc = "best")
        plt.title("Performance over episodes for move {}".format(i + 1))
        plt.savefig("./Report/images/7/move_dist_{}.png".format(i + 1))
        # plt.show()
    policy = Policy()
    load_weights(policy, 49500)
    final_move_dist = np.array(first_move_distr(policy, env))[0]
    print([float("{:.4f}".format(i)) for i in final_move_dist])
    return


def part8(env, policy):
    load_weights(policy, 49500)
    lose_count = 0
    while lose_count < 5:
        print("===== NEW GAME =====")
        state = env.reset()
        done = False
        status = None
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            env.render()
        if status == Environment.STATUS_LOSE:
            lose_count += 1
            print("{} lose {} {}".format("=" * 20, lose_count, "=" * 20))
    return


if __name__ == '__main__':
    import sys

    # Set seeds
    np.random.seed(59)
    random.seed(59)
    torch.manual_seed(59)

    # Initialize env and policy
    env = Environment()
    policy = Policy()

    # part1(env)
    # part2()
    # part3()
    # part4()
    # part5(env, policy, part_b = False)
    # part6(env)
    # part7(env)
    part8(env, policy)

    # if len(sys.argv) == 1:
    #     # `python tictactoe.py` to train the agent
    #     train(policy, env)
    # else:
    #     # `python tictactoe.py <ep>` to print the first move distribution
    #     # using weightt checkpoint at episode int(<ep>)
    #     ep = int(sys.argv[1])
    #     load_weights(policy, ep)
    #     print(first_move_distr(policy, env))
