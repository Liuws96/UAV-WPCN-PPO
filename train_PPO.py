#!/usr/bin/env python3
import sys

import wandb

sys.setrecursionlimit(3000)
import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, Beta
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import *
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic, ActorProbBeta

from main_env_WPCN_single import MainEnv

USE_BETA = False
num_users = 5
p_max = 1e-4
p_max_server = 20
E_users_max = 1e-5
f_max = 1e8
UAV_R = np.inf

Dmin = 1e5
Dmax = 2e5

WIDTH = 50
B = 1e6
N = 40

eta = 0.8
sigma2dB = -90
sigma2 = 10 ** (sigma2dB / 10)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='WPCN-PPO')
    parser.add_argument("--algorithm_name", type=str, default='PPO')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1e5)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 64])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--repeat-per-collect', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    # ppo special
    parser.add_argument('--rew-norm', type=int, default=False)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--lr-decay', type=int, default=False)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--resume-from-log', type=bool, default=False)  # 1
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    MODEL_DIR = None
    parser.add_argument('--resume-path', type=str, default=MODEL_DIR)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )

    parser.add_argument('--scenario_name', type=str, default='MainEnv', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)

    parser.add_argument('--num_users', type=int, default=num_users, help="number of users")
    parser.add_argument("--ground_length", type=float, default=WIDTH, help="ground length")
    parser.add_argument("--ground_width", type=float, default=WIDTH, help="ground width")
    parser.add_argument("--B", type=float, default=B, help="bandwidth")
    parser.add_argument("--p_max", type=float, default=p_max, help="transmit power")
    parser.add_argument("--p_max_server", type=float, default=p_max_server, help="transmit power")
    parser.add_argument("--p_min", type=float, default=0, help="transmit power")
    parser.add_argument("--sigma2", type=float, default=sigma2, help="noise power")
    parser.add_argument("--eta", type=float, default=eta, help="noise power")
    parser.add_argument("--E_users_max", type=float, default=E_users_max, help="noise power")

    parser.add_argument("--delta_T", type=float, default=0.2, help="time slot")
    parser.add_argument("--UAV_H", type=float, default=10, help="UAV height")

    parser.add_argument("--BS_H", type=float, default=10, help="BS height")
    parser.add_argument("--relay_H", type=float, default=500, help="BS height")

    parser.add_argument("--deg_users_min", type=float, default=0)
    parser.add_argument("--deg_users_max", type=float, default=2 * np.pi)
    parser.add_argument("--v_users_min", type=float, default=1)
    parser.add_argument("--v_users_max", type=float, default=4)
    parser.add_argument("--v_servers_max", type=float, default=15, help="maximum velocity of UAVs")
    parser.add_argument("--a_servers_max", type=float, default=3, help="maximum accelerate of UAVs")
    parser.add_argument("--v_users_noise_sigma2", type=float, default=1, help="maximum velocity of UAVs")
    parser.add_argument("--deg_users_noise_sigma2", type=float, default=1, help="maximum velocity of UAVs")
    parser.add_argument("--mu_v", type=float, default=0.2, help="maximum velocity of UAVs")
    parser.add_argument("--mu_deg", type=float, default=0.2, help="maximum velocity of UAVs")

    parser.add_argument("--f_max", type=float, default=f_max)
    parser.add_argument("--f_UAV", type=float, default=10e9)

    parser.add_argument("--p_hover", type=float, default=0.5, help="power")
    parser.add_argument("--RC", type=float, default=1e-28, help="power")
    parser.add_argument("--Dmin", type=int, default=Dmin, help="minimum data size")
    parser.add_argument("--Dmax", type=int, default=Dmax, help="maximum data size")
    parser.add_argument("--Cmin", type=int, default=0.5e3, help="maximum data size")
    parser.add_argument("--Cmax", type=int, default=1.5e3, help="maximum data size")

    parser.add_argument("--UAV_R", type=int, default=UAV_R, help="maximum data coverage")
    parser.add_argument("--user_R", type=int, default=UAV_R, help="maximum user coverage")

    parser.add_argument("--UAV_START_AT_O", type=bool, default=False, help="number of BSs.")
    parser.add_argument("--REW_CLIP", type=float, default=10, help="action cases")
    parser.add_argument("--E_FLY_WEIGHT", type=float, default=1, help="action cases")
    parser.add_argument("--USE_BETA", type=bool, default=USE_BETA, help="use beta distribution")
    parser.add_argument("--USE_ACCELERATE", type=bool, default=True, help="alpha fixed in")

    parser.add_argument("--DEMO", type=bool, default=False, help="DEMO")
    parser.add_argument("--env_name", type=str, default='', help="specify the name of environment")
    parser.add_argument("--experiment_name", type=str, default=f"K={num_users},B={B},BETA={USE_BETA},sigma2={sigma2dB},Dmax={Dmax},WIDTH={WIDTH},UAV_R={UAV_R},N={N},eta={eta},p_max={p_max},pe_max={p_max_server},f_max={f_max}", help="an identifier to distinguish different experiment.")
    parser.add_argument("--centralized_executing", type=bool, default=True, help="centralized_executing")
    parser.add_argument("--use_wandb", type=bool, default=True, help="use wandb")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=10, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,  # fake
                        default=N, help="Max length for any episode")

    model_dir = None  # files
    parser.add_argument("--model_dir", type=str, default=model_dir, help="by default None. set the path to pretrained model.")

    return parser.parse_args()


def train_ppo(args=get_args()):
    env = MainEnv(0, args)
    args.state_shape = env.observation_space.shape or env.observation_space.shape
    args.action_shape = env.action_space.shape or env.action_space.shape
    args.max_action = env.action_space.high

    args.is_long_decision_agent = 1
    args.ID = wandb.run.id if wandb.run is not None else None
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # print("Action range:", np.min(env.share_action_space[0].low), np.max(env.share_action_space[0].high))
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: MainEnv(_, args) for _ in range(args.training_num)], norm_obs=False
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: MainEnv(_, args) for _ in range(args.test_num)],
        norm_obs=False,
        obs_rms=train_envs.obs_rms,
        update_obs_rms=False
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    if not args.USE_BETA:
        actor = ActorProb(
            net_a,
            args.action_shape,
            max_action=args.max_action,
            unbounded=True,
            device=args.device
        ).to(args.device)
    else:
        actor = ActorProbBeta(
            net_a,
            args.action_shape,
            max_action=args.max_action,
            unbounded=True,
            device=args.device
        ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    if not args.USE_BETA:
        torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    if not args.USE_BETA:
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
    else:
        for m in actor.action_out.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight.data)
                torch.nn.init.zeros_(m.bias)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr
    )

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(*logits):
        return Independent(Normal(*logits), 1) if not args.USE_BETA else Beta(*logits)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    buffer_test = VectorReplayBuffer(args.episode_length, len(train_envs))  # 默认env_num大小

    def single_preprocess_fn(**kwargs):
        if 'rew' in kwargs:
            info = kwargs['info']
            return info
        else:
            return Batch()

    train_collector = Collector(policy, train_envs, buffer, preprocess_fn=single_preprocess_fn, exploration_noise=True)
    test_collector = Collector(policy, test_envs, buffer_test)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_ppo'
    log_path = os.path.join(args.logdir, args.task, 'ppo', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if not args.use_wandb:
        logger = TensorboardLogger(writer, update_interval=1, train_interval=10)
    else:
        logger = WandbLogger(
            name=str(args.algorithm_name) + "_" +
                 str(args.experiment_name) +
                 "_seed" + str(args.seed),
            run_id=args.model_dir.split("\\")[-2][-8:] if args.model_dir is not None else None,
            update_interval=args.episode_length,
            train_interval=args.episode_length,
            config=args,
        )

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    save_checkpoint_fn = None
    # def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
    #     # if epoch % args.save_interval == 0:
    #     #     save_fn(policy)
    #     return os.path.join(log_path, 'policy.pth')

    # args.step_per_collect = args.episode_length  # !!!!!!!!!!!!!!!
    if not args.watch:
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            resume_from_log=args.resume_from_log,
            save_checkpoint_fn=save_checkpoint_fn,
            save_fn=save_fn,
            stop_fn=lambda x: False,
            logger=logger,
            test_in_train=True
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    train_ppo()
