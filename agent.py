import argparse
import random
import flappy_bird_gymnasium
import gymnasium as gym
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory
import os
import itertools


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):

        self.param_set = param_set

        with open("parameter.yaml", "r") as f:
            all_params_set = yaml.safe_load(f)
            params = all_params_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )


    def run(self, is_training=True, render=False):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        if is_training:

            memory = ReplayMemory(
                maxlen=self.replay_memory_size
            )

            epsilon = self.epsilon_init

            target_dqn = DQN(
                num_states,
                num_actions
            ).to(device)

            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

            steps = 0

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            best_reward = -float("inf")

        else:

            policy_dqn.load_state_dict(
                torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )
            )

            policy_dqn.eval()

        for episode in itertools.count():

            state, _ = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float,
                device=device
            )

            episode_reward = 0
            terminated = False

            while (
                not terminated
                and episode_reward < self.reward_threshold
            ):

                if (
                    is_training
                    and random.random() < epsilon
                ):

                    action = env.action_space.sample()

                    action = torch.tensor(
                        action,
                        dtype=torch.long,
                        device=device
                    )

                else:

                    with torch.no_grad():

                        action = (
                            policy_dqn(
                                state.unsqueeze(dim=0)
                            )
                            .squeeze()
                            .argmax()
                        )

                next_state, reward, terminated, _, _ = env.step(
                    action.item()
                )

                episode_reward += reward

                reward = torch.tensor(
                    reward,
                    dtype=torch.float,
                    device=device
                )

                next_state = torch.tensor(
                    next_state,
                    dtype=torch.float,
                    device=device
                )

                if is_training:

                    memory.append(
                        (
                            state,
                            action,
                            next_state,
                            reward,
                            terminated
                        )
                    )

                    steps += 1

                state = next_state

            if is_training:

                print(
                    f"Episode {episode} finished "
                    f"with reward {episode_reward} "
                    f"and epsilon {epsilon}"
                )

                epsilon = max(
                    self.epsilon_min,
                    epsilon * self.epsilon_decay
                )

                if episode_reward > best_reward:

                    log_msg = (
                        f"New best reward "
                        f"{episode_reward} "
                        f"at episode {episode + 1}\n"
                    )

                    with open(
                        self.LOG_FILE,
                        "a"
                    ) as log_file:

                        log_file.write(log_msg)

                    best_reward = episode_reward

                    torch.save(
                        policy_dqn.state_dict(),
                        self.MODEL_FILE
                    )

                    print(
                        f"New best reward {best_reward}. "
                        f"Model saved to {self.MODEL_FILE}"
                    )

                if len(memory) > self.mini_batch_size:

                    mini_batch = memory.sample(
                        self.mini_batch_size
                    )

                    self.optimize(
                        mini_batch,
                        policy_dqn,
                        target_dqn
                    )

                    if steps >= self.network_sync_rate:

                        target_dqn.load_state_dict(
                            policy_dqn.state_dict()
                        )

                        steps = 0

            else:

                print(
                    f"Episode {episode} "
                    f"finished with reward {episode_reward}"
                )


        env.close()


    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn
    ):

        # Get experiences

        states, actions, next_states, rewards, dones = zip(
            *mini_batch
        )

        states = torch.stack(states)

        actions = torch.stack(actions)

        next_states = torch.stack(next_states)

        rewards = torch.stack(rewards)

        dones = torch.tensor(
            dones,
            dtype=torch.float,
            device=device
        )


        # Compute Q values for next states

        with torch.no_grad():

            target_q = (
                rewards
                + (1 - dones)
                * self.gamma
                * target_dqn(next_states)
                .max(dim=1)[0]
            )


        # Compute Q values for current states

        current_q = (
            policy_dqn(states)
            .gather(
                1,
                actions.unsqueeze(dim=1)
            )
            .squeeze()
        )


        # Compute loss

        loss = self.loss_fn(
            current_q,
            target_q
        )


        # Backpropagation

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or test model."
    )

    parser.add_argument(
        "hyperparameters",
        help=""
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )

    args = parser.parse_args()

    dql = Agent(
        args.hyperparameters
    )

    if args.train:

        dql.run(
            is_training=True
        )

    else:

        dql.run(
            is_training=False,
            render=True
        )


# import argparse
# import random

# import flappy_bird_gymnasium
# import gymnasium as gym
# import yaml
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dqn import DQN
# from experience_replay import ReplayMemory 
# import os 
# import itertools
# if torch.cuda.is_available():
#     device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cpu"

# RUNS_DIR = "runs"
# os.makedirs(RUNS_DIR, exist_ok=True)

# class Agent:

#     def __init__(self,param_set):
#         self.param_set=param_set
#         with open("parameters.yaml", "r") as f:
#             all_params_set = yaml.safe_load(f)
#             params=all_params_set[param_set]
#         self.alpha=params["alpha"]
#         self.gamma=params["gamma"]
#         self.epsilon_init=params["epsilon_init"]
#         self .epsiolon_decay=params["epsilon_decay"]
#         self.epsilon_min=params["epsilon_min"]
#         self.replay_memory_size=params["replay_memory_size"]
#         self.mini_batch_size=params["mini_batch_size"]

#         self.reward_threshold=params["reward_threshold"]
#         self.network_sync_rate=params["network_sync_rate"]
#         self.mini_batch_size=params["mini_batch_size"]

#         self.loss_fn=nn.MSELoss()
#         self.optimizer=None
#         self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
#         self.MODEL_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.pt")
#     def run(self,is_training=True,render=False) :
    
#         env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

#         num_states=env.observation_space.shape[0] #input dimension

#         num_actions=env.action_space.n  # output dimension
#         policy_dqn=DQN(num_states,num_actions)
#         state, _ = env.reset()
#         if is_training:
#             memory=ReplayMemory(maxlen=self.replay_memory_size)
#             epsilon=self.epsilon_init
#             target_dqn=DQN(num_states,num_actions).to(device)
#             target_dqn.load_state_dict(policy_dqn.state_dict())
#             steps=0;
#             self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
#             best_reward=-float('inf')
#         else :
#             policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
#             policy_dqn.eval()


#         for episode in itertools.count():
#             state, _ = env.reset()
#             state=torch.tensor(state,dtype=torch.float,device=device)
#             episode_reward = 0
#             terminated = False
#             while not terminated and  episode_reward < self.reward_threshold:
#                 if is_training and random.random() < epsilon:
#                     action = env.action_space.sample()  # Explore action space
#                     action = torch.tensor(action,dtype=torch.long,device=device)
#                 else:
#                     with torch.no_grad():
#                         action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
#                 # Processing: terminated =>done
#                 next_state, reward, terminated, _, _ = env.step(action.item())
#                 episode_reward += reward
#                 reward = torch.tensor(reward, dtype=torch.float, device=device)
#                 next_state = torch.tensor(next_state, dtype=torch.float, device=device)
#                 if is_training:
#                     memory.append((state, action, next_state,reward, terminated))
#                     steps+=1

#                 state = next_state
                
#             print(f"Episode {episode} finished with reward {episode_reward} and epsilon {epsilon}")
#             if is_training:

#                 epsilon = max(self.epsilon_min, epsilon * self.epsiolon_decay)
#                 if episode_reward > best_reward:
#                     log_msg = f"New best reward {episode_reward} at episode {episode+1}\n"
#                     with open(self.LOG_FILE, "a") as log_file:
#                         log_file.write(log_msg)
#                     best_reward = episode_reward
#                     torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
#                     print(f"New best reward {best_reward}. Model saved to {self.MODEL_FILE}")
#             if is_training and len(memory) > self.mini_batch_size:
#                 mini_batch=memory.sample(self.mini_batch_size)
#                 self.optimize(mini_batch,policy_dqn,target_dqn)
#                 if(steps>self.network_sync_rate):
#                     target_dqn.load_state_dict(policy_dqn.state_dict())
#                     steps=0


#     def optimize(self, mini_batch, policy_dqn, target_dqn):
#         # get experiences
#         states, actions, next_states, rewards, dones = zip(*mini_batch)
#         states = torch.stack(states)
#         actions = torch.stack(actions)
#         next_states = torch.stack(next_states)
#         rewards = torch.stack(rewards)
#         terminations=torch.tensor(terminations).float().to(device)
#         # Compute Q values for current states
#         with torch.no_grad():
#             target_q=rewards+(1-terminations)*self.gamma*target_dqn(next_states).max(dim=1)[0]
#         current_q=policy_dqn(states).gather(1,actions.unsqueeze(dim=1)).squeeze()
#         loss = self.loss_fn(current_q, target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# if __name__ == "__main__":
#     parser=argparse.ArgumentParser(description="Train or test model.")
#     parser.add_argument('hyperparameters',help='')
#     parser.add_argument('--train',action='store_true',help='Train the model')
#     args=parser.parse_args()
#     dql=Agent(args.hyperparameters)
#     if args.train:
#         dql.run(is_training=True)
#     else:
#         dql.run(is_training=False,render=True)
    


