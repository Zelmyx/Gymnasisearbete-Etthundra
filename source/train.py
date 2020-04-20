import os
from random import randint
from RL_Brain import SarsaTable


def print_state(player1_score, player2_score, temp_score, turn):
    os.system("cls" if os.name == "nt" else "clear")

    print(f"Player1 score: {player1_score}")
    print(f"Player2 score: {player2_score}")

    if turn % 2 == 1:
        player = "Player1"
    else:
        player = "Player2"

    print(f"{player}s temporary score is: {temp_score}")


def main():
    player1_score = 0
    player2_score = 0
    temp_score = 0
    game_over = False
    turn = 1
    score_to_win = 100

    while not game_over:

        if turn % 2 == 1:
            AI = AI1
        else:
            AI = AI2

        observation = str([player1_score, player2_score, temp_score])

        # Choose action based on observation
        yes_or_no = AI.choose_action(observation)

        if yes_or_no == "'":
            outcome = randint(1, 6)

            if outcome == 1:
                temp_score = 0
                turn += 1
            else:
                temp_score += outcome
        else:
            reward = 0
            reward1 = 0
            reward2 = 0

            if turn % 2 == 1:
                player1_score += temp_score
                if player1_score >= score_to_win:

                    reward1 = 100
                    reward2 = -100
                    game_over = True

                    observation_ = str(
                                    [player1_score, player2_score, temp_score]
                                   )
                    action_ = AI1.choose_action(observation_)
                    # Learn from this transition (s, a, r, s, a) --> SARSA
                    AI1.learn(observation, yes_or_no, reward1,
                              observation_, action_, game_over)
                    AI2.learn(temp_observation, temp_action, reward2,
                              temp_observation, temp_action, game_over)
            else:
                player2_score += temp_score
                if player2_score >= score_to_win:

                    reward1 = -100
                    reward2 = 100
                    game_over = True

                    observation_ = str([player1_score, player2_score, temp_score])
                    action_ = AI2.choose_action(observation_)
                    # Learn from this transition (s, a, r, s, a) --> SARSA
                    AI2.learn(observation, yes_or_no, reward2, observation_, action_, game_over)
                    AI1.learn(temp_observation, temp_action, reward1, temp_observation, temp_action, game_over)

            if not game_over:
                observation_ = str([player1_score, player2_score, temp_score]) 

                action_ = AI.choose_action(observation_)
                # Learn from this transition (s, a, r, s, a) --> SARSA
                AI.learn(observation, yes_or_no, reward, observation_, action_, game_over)

                temp_observation = observation
                temp_action = yes_or_no

                # Swap observation and action
                observation = observation_
                yes_or_no = action_

                temp_score = 0
                turn += 1


try:
    with open("progress.txt", "r") as f:
        num_games_already_played = f.read()
        num_games_already_played = int(num_games_already_played)
except FileNotFoundError:
    num_games_already_played = 1

try:
    with open("epsilon_decay", "r") as f:
        e_greedy = float(f.read())
except FileNotFoundError:
    e_greedy = 0.9

checkpoints = []
start = 100
for _ in range(6):
    for i in range(1, 10):
        checkpoints.append(i*start)
    start *= 10

for checkpoint in checkpoints:
    if checkpoint < num_games_already_played:
        checkpoints.remove(checkpoint)

games_to_train = 100_000_000

if __name__ == "__main__":
    AI1 = SarsaTable(name="RL_1", actions=["n", "'"], e_greedy=e_greedy)
    AI2 = SarsaTable(name="RL_2", actions=["n", "'"], e_greedy=e_greedy)
    print(f"Games played: {num_games_already_played}")

    for i in range(num_games_already_played, games_to_train + 1):
        try:
            main()
            if i in checkpoints:
                AI1.save()
                AI2.save()
                AI1.save_stage(i)
                AI2.save_stage(i)
                print(f"Saved at {i}")
            if i % 1000 == 0:
                AI1.epsilon_decay()
                AI2.epsilon_decay()
        except KeyboardInterrupt as e:
            AI1.save()
            AI2.save()
            AI1.save_progress(i)
            raise e

    AI1.save()
    AI2.save()
    print("Done")
