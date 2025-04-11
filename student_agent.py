from run_game import NTupleApproximator, Game2048Env, TD_MCTS, TD_MCTS_Node

patterns = [
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
]

env = Game2048Env()
approximator = NTupleApproximator(board_size=4, patterns=patterns)
td_mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, V_norm=80000)

def get_action(state, score):
    env.board = state.copy()
    env.score = score
    root = TD_MCTS_Node(state)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    action = td_mcts.best_action(root)
    return action


def main(num_episodes=5):
    final_scores = []
    
    env = Game2048Env()
    for episode in range(num_episodes):
        env.reset()
        done = False

        while not done:
            action = get_action(env.board, env.score)
            _, current_score, done, _ = env.step(action)

        print(f"Game {episode + 1} completed! Score: ", env.score)
        final_scores.append(env.score)

    print("Average Score: ", sum(final_scores) / len(final_scores))
    return final_scores

if __name__ == "__main__":
    main()
