import os
import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, name, action_space, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.9):

        self.actions = ["n", "'"]   # The possible moves
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.name = name

        if self.name == "RL_1":
            try:
                self.q_table = pd.read_pickle("q_table_1.pk")
            except FileNotFoundError:
                self.q_table = pd.DataFrame(columns=self.actions,
                                            dtype=np.float32)
        elif self.name == "RL_2":
            try:
                self.q_table = pd.read_pickle("q_table_2.pk")
            except FileNotFoundError:
                self.q_table = pd.DataFrame(columns=self.actions,
                                            dtype=np.float32)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Append new state to q-table
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),
                                               index=self.q_table.columns,
                                               name=state))

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() > self.epsilon:
            # Choose best action
            state_action = self.q_table.loc[observation, :]
            # Some actions may have the same value,
            # randomly choose on in these actions
            action = np.random.choice(
                            state_action[
                                state_action == np.max(state_action)
                            ].index)
        else:
            # Choose random action
            action = np.random.choice(self.actions)
        return action

    def epsilon_decay(self):
        self.epsilon *= 0.9999

    def save_epsilon_decay(self):
        with open("epsilon_decay.txt", "w") as f:
            f.write(str(self.epsilon))


class SarsaTable(RL):

    def __init__(self, name, actions, learning_rate=0.01, 
                 reward_decay=0.9, e_greedy=0.9):

        super(SarsaTable, self).__init__(name, actions, learning_rate,
                                         reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_, terminal=False):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if terminal is not True:  # Next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r  # Next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # Update

    def save(self):
        if self.name == "RL_1":
            self.q_table.to_pickle("q_table_1.pk")
        elif self.name == "RL_2":
            self.q_table.to_pickle("q_table_2.pk")
        self.save_epsilon_decay()

    def save_stage(self, num_games):
        if os.name == "nt":
            copy_command = "copy"
            move_command = "move"
            slash = "\\"
        else:
            copy_command = "cp"
            move_command = "mv"
            slash = "/"

        if self.name == "RL_1":
            os.system(f"{copy_command} q_table_1.pk Results{slash}")
            filename = f"{num_games}_player_1.pk"
            os.system(f"{move_command} Results{slash}q_table_1.pk Results{slash}{filename}")
        elif self.name == "RL_2":
            os.system(f"{copy_command} q_table_2.pk Results{slash}")
            filename = f"{num_games}_player_2.pk"
            os.system(f"{move_command} Results{slash}q_table_2.pk Results{slash}{filename}")

    @staticmethod
    def save_progress(num_games):
        with open(f"progress.txt", "w") as f:
            f.write(str(num_games))
