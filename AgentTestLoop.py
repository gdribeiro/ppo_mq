import argparse
import numpy as np
from agent import createPPOAgent, infer, finish

def main(qosmin, qosmax):
    start_state = [0] * 8
    print(f"Start state: {start_state}")

    agent = createPPOAgent(start_state, qosmin, qosmax)

    for _ in range(5):
        print("#" * 20)
        print(f"Step: {_}")
        print("#" * 20)
        
        random_state = np.random.randint(0, 100, size=8).tolist()
        print(f"Random state: {random_state}")
        action = infer(agent, random_state)
        print(f"Action taken: {action}")

    random_state = np.random.randint(0, 100, size=8).tolist()
    finish(agent, final_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO Agent")
    parser.add_argument("--qosmin", type=int, default=1, help="Minimum QoS")
    parser.add_argument("--qosmax", type=int, default=32, help="Maximum QoS")
    args = parser.parse_args()

    main(args.qosmin, args.qosmax)