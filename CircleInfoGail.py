import math
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pathlib
import torch

class CircleEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left

    The CircleEnv class inherits from gym.Env which is a class that
    creates an interface to basic gym environment rules and dynamics.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self):
        '''
        There are two variables the action_space, which contains 2
        elements that goes each from -10 to 10 and the observation_space
        which contains 10 elements (the pair position of the five last 
        positions) that goes from -10 to 10, as well. Bothe variables
        are np.float32.
        '''
        super(CircleEnv, self).__init__()
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)

    def reset(self):
        self.center_x = 0.0
        self.center_y = 0.3
        self.episode_reward = 0
        self.speed = 0.01
        self.angle = 0.0
        self.cur_length = 0
        self.ep_length = 1000
        self.hist5 = []
        for _ in range(5):
            pos_x = np.random.normal(scale=0.001)
            pos_y = np.random.normal(scale=0.001)
            self.hist5.append(pos_x)
            self.hist5.append(pos_y)
        
        return np.array(self.hist5).astype(np.float64)
    
    def step(self, action):
        ''' 
        Apply the new position given the action, updates the hist5 list
        and computes the reward.

        hist5 is a column list with 10 rows and 1 column.
        [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]

        The action is a list with two rows and 1 column.
        [x, y]

        pos_x = pos_x + speed * cos + noise
        pos_y = pos_y + speed + sin + noise
        '''
        pos_x = self.hist5[-2]
        pos_y = self.hist5[-1]
        action_angle = math.atan2(action[1], action[0])
        pos_x += self.speed * math.cos(action_angle)
        pos_y += self.speed * math.sin(action_angle)
        pos_x += np.random.normal(scale=0.001)
        pos_y += np.random.normal(scale=0.001)

        for i in range(0, len(self.hist5) - 2):
            self.hist5[i] = self.hist5[i+2]
            self.hist5[-2] = pos_x
            self.hist5[-1] = pos_y
        done = False
        self.cur_length += 1
        reward = -abs((self.center_x ** 2 + self.center_y ** 2) ** 0.5 - ((pos_x - self.center_x) ** 2 + (pos_y - self.center_y) ** 2) ** 0.5
                      )/(self.center_x ** 2 + self.center_y ** 2)
        self.episode_reward += reward
        info = {}
        if self.cur_length >= self.ep_length:
            info['episode'] = {'r': self.episode_reward}
            done = True

        return np.array(self.hist5.copy()).astype(np.float64), reward, done, info
    
    def close(self):
        pass


def circle_expert(obs):
    center_x = 0.0
    center_y = 0.3
    pos_x = obs[-2]
    pos_y = obs[-1]
    action_angle = math.atan2(pos_y - center_y, pos_x - center_x) + math.pi / 2
    radius_factor = ((center_x ** 2 + center_y ** 2) ** 0.5 - ((pos_x - center_x) ** 2 + (pos_y - center_y) ** 2) ** 0.5
                    )/(center_x ** 2 + center_y ** 2) ** 0.5
    action_angle -= 3 * radius_factor

    return [math.cos(action_angle), math.sin(action_angle)]

def gen_trajectories():
    # Instatiate the env
    env = CircleEnv()

    # Test the trained agent
    n_episodes = 20
    n_steps = 1000
    obs = env.reset()
    states = []
    actions = []

    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(n_steps):
            states.append(obs)
            action = circle_expert(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            if done:
                break

    states = torch.as_tensor(states).float()
    actions = torch.as_tensor(actions).float()
    perm = torch.randperm(states.shape[0])
    states = states[perm]
    actions = actions[perm]
    data = {
        'states': states,
        'actions': actions,
    }
    return data

def evaluate_policy(id, model):
    n_steps = 1000
    pos_hist = [[], []]
    env = CircleEnv()
    obs = env.reset()
    for _ in range(n_steps):
        pos_hist[0].append(obs[-2])
        pos_hist[1].append(obs[-1])
        action = model(torch.as_tensor(obs).float())
        obs, _, done, _ = env.step(action)
        if done:
            break
    plt.scatter(*pos_hist)

    plt.title('Trajectory {}'.format(id))
    plt.show()
    plt.clf

eval_path = pathlib.PosixPath('eval')
if not eval_path.exists():
    pathlib.Path.mkdir(eval_path)

evaluate_policy('expert', circle_expert)
trajectories = gen_trajectories()

hidden_size = 32

model = torch.nn.Sequential(
    torch.nn.Linear(trajectories['states'][0].shape[0], hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, trajectories['actions'][0].shape[0]),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 32
n_steps_per_epoch = len(trajectories['states']) // batch_size
n_epochs = 10
i_eval = 0
loss_hist = []
for epoch in range(n_epochs):
    for t in range(n_steps_per_epoch):
        # Forward pass: compute predicted y by passing x to the model
        i_start = t* batch_size
        i_end = (t+1)*batch_size
        action_pred = model(trajectories['states'][i_start:i_end])

        # Compute and print loss.
        loss = loss_fn(action_pred, trajectories['actions'][i_start:i_end])
        if t % 100 == 99:
            evaluate_policy(i_eval, model)
            i_eval += 1
        
        loss_hist.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

plt.plot(loss_hist)
plt.title('Loss history')
plt.show()
plt.clf()