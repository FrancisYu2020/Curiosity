from matplotlib.pyplot import step
import torch
import numpy as np
from PIL import Image

def test_model_in_env(model, env, episode_len, device, vis=False, vis_save=False):
    g = 0
    state = env.reset()
    init_state = state
    gif = []
    
    with torch.no_grad():
        for t in range(episode_len):
            state = torch.from_numpy(state).unsqueeze(0).float().to(device)
            action = model.act(state).detach().cpu().numpy()
            state, reward, done, info = env.step(action[0])
            g += reward
            if vis: env.render()
            if vis_save: gif.append(Image.fromarray(env.render(mode='rgb_array')))
            if done: break
    return state, g, gif, info

def val(model, device, envs, episode_len):
    states = [e.reset() for e in envs]
    all_rewards = []
    dones = [False for _ in envs]
    for i in range(episode_len):
        arr = []
        for ent in states:
            arr.append(np.asarray(ent))
        arr = np.array(arr)
        states = arr

        # states = np.array(states)
        with torch.no_grad():
            _states = torch.from_numpy(states).float().to(device)
            _actions = model.act(_states)
        actions = _actions.cpu().numpy()
        step_data = []
        for i, env in enumerate(envs):
            if dones[i]:
                s = torch.from_numpy(np.array([env.reset()])).to(device)
                action = model.act(s).cpu().numpy()
                step_data.append(env.step(action[0]))
                # if len(step_data) == i + 1:
                #     action = model.act(s).cpu().numpy()
                #     step_data[-1] = env.step(action[0])
                # else:
                #     step_data.append(env.step(actions[i]))
                # dones[i] = step_data[-1][-2]
            else:
                step_data.append(env.step(actions[i]))
            # dones[i] = step_data[-1][-2]
        # print(step_data)
        # step_data = [env.step(actions[i]) for i, env in enumerate(envs)]
        [env.render() for env in envs]
        new_states, rewards, dones, infos = list(zip(*step_data))
        states = new_states
        all_rewards.append(rewards)
    all_rewards = np.array(all_rewards)
    return np.mean(np.sum(all_rewards, 0))
