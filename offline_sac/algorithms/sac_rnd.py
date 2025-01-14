import os
import wandb
import uuid
import pyrallis

import jax
import numpy as np
import jax.numpy as jnp
import optax

from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Optional, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState

from offline_sac.networks import RND, Actor, EnsembleCritic, Alpha
from offline_sac.utils.buffer import ReplayBuffer
from offline_sac.utils.common import Metrics, make_env, evaluate
from offline_sac.utils.running_moments import RunningMeanStd


@dataclass
class Config:
    # wandb params
    project: str = "SAC-RND"
    group: str = "sac-rnd"
    name: str = "sac-rnd"
    # model params
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    beta: float = 1.0
    num_critics: int = 2
    critic_layernorm: bool = True
    # rnd params
    rnd_learning_rate: float = 3e-4
    rnd_hidden_dim: int = 256
    rnd_embedding_dim: int = 32
    rnd_mlp_type: str = "concat_first"
    rnd_target_mlp_type: Optional[str] = None
    rnd_switch_features: bool = True
    rnd_update_epochs: int = 500
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


class RNDTrainState(TrainState):
    rms: RunningMeanStd  # 用于跟踪奖励的均值和标准差

class CriticTrainState(TrainState):
    target_params: FrozenDict  # 评论家的目标网络参数

# RND 函数
def rnd_bonus(
        rnd: RNDTrainState,  # RND 训练状态
        state: jax.Array,    # 当前状态
        action: jax.Array    # 当前动作
) -> jax.Array:
    # 通过 RND 模型预测和目标网络计算
    pred, target = rnd.apply_fn(rnd.params, state, action)
    # 计算 RND 奖励（bonus），即预测和目标之间的均方误差，归一化为标准差
    bonus = jnp.sum((pred - target) ** 2, axis=1) / rnd.rms.std
    return bonus

def update_rnd(
        key: jax.random.PRNGKey,  # 随机数种子
        rnd: RNDTrainState,        # RND 训练状态
        batch: Dict[str, jax.Array],  # 批量数据，包括状态和动作
        metrics: Metrics            # 记录指标的对象
) -> Tuple[jax.random.PRNGKey, RNDTrainState, Metrics]:
    def rnd_loss_fn(params):
        # 计算 RND 的损失
        pred, target = rnd.apply_fn(params, batch["states"], batch["actions"])
        raw_loss = ((pred - target) ** 2).sum(axis=1)  # 原始损失
        new_rms = rnd.rms.update(raw_loss)  # 更新均值和标准差
        loss = raw_loss.mean(axis=0)  # 计算平均损失
        return loss, new_rms

    # 计算损失和梯度
    (loss, new_rms), grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(rnd.params)
    new_rnd = rnd.apply_gradients(grads=grads).replace(rms=new_rms)  # 更新 RND 状态

    # 记录随机动作的 RND 奖励
    key, actions_key = jax.random.split(key)
    random_actions = jax.random.uniform(actions_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
    new_metrics = metrics.update({
        "rnd_loss": loss,
        "rnd_rms": new_rnd.rms.std,
        "rnd_data": loss / rnd.rms.std,
        "rnd_random": rnd_bonus(rnd, batch["states"], random_actions).mean()
    })
    return key, new_rnd, new_metrics

def update_actor(
        key: jax.random.PRNGKey,  # 随机数种子
        actor: TrainState,         # 演员的训练状态
        rnd: RNDTrainState,       # RND 训练状态
        critic: TrainState,       # 评论家的训练状态
        alpha: TrainState,        # 温度参数
        batch: Dict[str, jax.Array],  # 批量数据
        beta: float,              # RND 奖励的权重
        metrics: Metrics          # 记录指标的对象
) -> Tuple[jax.random.PRNGKey, TrainState, jax.Array, Metrics]:
    key, actions_key, random_action_key = jax.random.split(key, 3)

    def actor_loss_fn(params):
        # 计算演员的动作分布
        actions_dist = actor.apply_fn(params, batch["states"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=actions_key)  # 采样动作和对数概率
        rnd_penalty = rnd_bonus(rnd, batch["states"], actions)  # 计算 RND 奖励
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)  # 计算评论家的 Q 值
        # 计算演员的损失
        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) + beta * rnd_penalty - q_values).mean()
        
        # 记录熵和其他指标
        actor_entropy = -actions_logp.sum(-1).mean()
        random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
        new_metrics = metrics.update({
            "batch_entropy": actor_entropy,
            "actor_loss": loss,
            "rnd_policy": rnd_penalty.mean(),
            "rnd_random": rnd_bonus(rnd, batch["states"], random_actions).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean()
        })
        return loss, (actor_entropy, new_metrics)

    # 计算损失和梯度
    grads, (actor_entropy, new_metrics) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)  # 更新演员状态
    return key, new_actor, actor_entropy, new_metrics

def update_alpha(
        alpha: TrainState,          # 温度参数的训练状态
        entropy: jax.Array,        # 当前熵值
        target_entropy: float,     # 目标熵值
        metrics: Metrics            # 记录指标的对象
) -> Tuple[TrainState, Metrics]:
    def alpha_loss_fn(params):
        alpha_value = alpha.apply_fn(params)  # 获取当前的 alpha 值
        loss = (alpha_value * (entropy - target_entropy)).mean()  # 计算损失
        new_metrics = metrics.update({
            "alpha": alpha_value,
            "alpha_loss": loss
        })
        return loss, new_metrics

    # 计算梯度并更新 alpha
    grads, new_metrics = jax.grad(alpha_loss_fn, has_aux=True)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)
    return new_alpha, new_metrics

def update_critic(
        key: jax.random.PRNGKey,  # 随机数种子
        actor: TrainState,         # 演员的训练状态
        rnd: RNDTrainState,       # RND 训练状态
        critic: CriticTrainState, # 评论家的训练状态
        alpha: TrainState,        # 温度参数
        batch: Dict[str, jax.Array],  # 批量数据
        gamma: float,             # 折扣因子
        beta: float,              # RND 奖励的权重
        tau: float,               # 目标网络更新的软更新参数
        metrics: Metrics          # 记录指标的对象
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key)
    next_actions_dist = actor.apply_fn(actor.params, batch["next_states"])  # 计算下一状态的动作分布
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=actions_key)  # 采样下一状态的动作
    rnd_penalty = rnd_bonus(rnd, batch["next_states"], next_actions)  # 计算 RND 奖励
    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)  # 计算下一状态的 Q 值
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1) - beta * rnd_penalty  # 计算目标 Q 值
    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q  # 计算目标 Q 值

    def critic_loss_fn(critic_params):
        # 计算当前状态的 Q 值
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()  # 取最小 Q 值的平均
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)  # 计算损失
        return loss, q_min

    # 计算损失和梯度
    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)  # 更新评论家状态
    new_critic = new_critic.replace(
        target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)  # 更新目标网络
    )
    new_metrics = metrics.update({
        "critic_loss": loss,
        "q_min": q_min,
    })
    return key, new_critic, new_metrics


def update_sac(
        key: jax.random.PRNGKey,  # 随机数种子
        rnd: RNDTrainState,        # RND 训练状态
        actor: TrainState,         # 演员的训练状态
        critic: CriticTrainState, # 评论家的训练状态
        alpha: TrainState,        # 温度参数的训练状态
        batch: Dict[str, Any],    # 批量数据
        target_entropy: float,     # 目标熵值
        gamma: float,              # 折扣因子
        beta: float,               # RND 奖励的权重
        tau: float,                # 目标网络更新的软更新参数
        metrics: Metrics           # 记录指标的对象
):
    # 更新演员并获取新的演员状态和熵
    key, new_actor, actor_entropy, new_metrics = update_actor(key, actor, rnd, critic, alpha, batch, beta, metrics)
    
    # 更新 alpha 参数
    new_alpha, new_metrics = update_alpha(alpha, actor_entropy, target_entropy, new_metrics)
    
    # 更新评论家并获取新的评论家状态
    key, new_critic, new_metrics = update_critic(
        key, new_actor, rnd, critic, alpha, batch, gamma, beta, tau, new_metrics
    )
    
    return key, new_actor, new_critic, new_alpha, new_metrics  # 返回更新后的状态和指标

def action_fn(actor: TrainState) -> Callable:
    @jax.jit  # JIT 编译加速
    def _action_fn(obs: jax.Array) -> jax.Array:
        # 根据观察值计算动作分布
        dist = actor.apply_fn(actor.params, obs)
        action = dist.mean()  # 取动作分布的均值作为最终动作
        return action
    return _action_fn  # 返回动作计算函数

@pyrallis.wrap()  # 使用 Pyrallis 包装配置
def main(config: Config):
    # 初始化 Weights & Biases（WandB）用于实验跟踪
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),  # 生成唯一 ID
    )
    
    # 创建重放缓冲区
    buffer = ReplayBuffer.create_from_d4rl(config.dataset_name, config.normalize_reward)
    state_mean, state_std = buffer.get_moments("states")  # 获取状态的均值和标准差
    action_mean, action_std = buffer.get_moments("actions")  # 获取动作的均值和标准差
    
    # 初始化随机数种子
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, rnd_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)  # 分割随机数种子
    
    # 创建评估环境
    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    init_state = buffer.data["states"][0][None, ...]  # 初始化状态
    init_action = buffer.data["actions"][0][None, ...]  # 初始化动作
    target_entropy = -init_action.shape[-1]  # 目标熵值
    
    # 初始化 RND 模块
    rnd_module = RND(
        hidden_dim=config.rnd_hidden_dim,
        embedding_dim=config.rnd_embedding_dim,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        mlp_type=config.rnd_mlp_type,
        target_mlp_type=config.rnd_target_mlp_type,
        switch_features=config.rnd_switch_features
    )
    
    # 创建 RND 训练状态
    rnd = RNDTrainState.create(
        apply_fn=rnd_module.apply,
        params=rnd_module.init(rnd_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.rnd_learning_rate),  # 使用 Adam 优化器
        rms=RunningMeanStd.create()  # 创建均值和标准差跟踪器
    )
    
    # 初始化演员模块
    actor_module = Actor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),  # 使用 Adam 优化器
    )
    
    # 初始化 alpha 模块
    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)  # 使用 Adam 优化器
    )
    
    # 初始化评论家模块
    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim, num_critics=config.num_critics, layernorm=config.critic_layernorm
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),  # 初始化目标参数
        tx=optax.adam(learning_rate=config.critic_learning_rate),  # 使用 Adam 优化器
    )
    
    # 部分应用 update_sac 函数，固定一些参数
    update_sac_partial = partial(
        update_sac, target_entropy=target_entropy, gamma=config.gamma, beta=config.beta, tau=config.tau
    )
    
    # RND 更新步骤
    def rnd_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])  # 分割随机数种子
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)  # 从缓冲区采样批量数据
        key, new_rnd, new_metrics = update_rnd(key, carry["rnd"], batch, carry["metrics"])  # 更新 RND
        carry.update(
            key=key, rnd=new_rnd, metrics=new_metrics  # 更新携带的状态
        )
        return carry
    
    # SAC 更新步骤
    def sac_loop_update_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])  # 分割随机数种子
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)  # 从缓冲区采样批量数据
        key, new_actor, new_critic, new_alpha, new_metrics = update_sac_partial(
            key=key,
            rnd=carry["rnd"],
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
            metrics=carry["metrics"]
        )  # 更新 SAC
        carry.update(
            key=key, actor=new_actor, critic=new_critic, alpha=new_alpha, metrics=new_metrics  # 更新携带的状态
        )
        return carry
    
    # 指标记录
    rnd_metrics_to_log = [
        "rnd_loss", "rnd_rms", "rnd_data", "rnd_random"
    ]
    bc_metrics_to_log = [
        "critic_loss", "q_min", "actor_loss", "batch_entropy",
        "rnd_policy", "rnd_random", "action_mse", "alpha_loss", "alpha"
    ]
    
    # 更新循环的共享携带状态
    update_carry = {
        "key": key,
        "actor": actor,
        "rnd": rnd,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }
    
    # 预训练 RND
    for epoch in range(config.rnd_update_epochs):
        # 每个 epoch 的指标累积和记录
        update_carry["metrics"] = Metrics.create(rnd_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=rnd_loop_update_step,
            init_val=update_carry
        )
        # 记录每个指标的平均值
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"RND/{k}": v for k, v in mean_metrics.items()}})
    
    # 训练 BC
    for epoch in range(config.num_epochs):
        # 每个 epoch 的指标累积和记录
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=sac_loop_update_step,
            init_val=update_carry
        )
        # 记录每个指标的平均值
        mean_metrics = update_carry["metrics"].compute()
        wandb.log({"epoch": epoch, **{f"SAC/{k}": v for k, v in mean_metrics.items()}})
        
        # 每隔一定的 epoch 进行评估
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            actor_action_fn = action_fn(actor=update_carry["actor"])  # 获取动作函数
            eval_returns = evaluate(eval_env, actor_action_fn, config.eval_episodes, seed=config.eval_seed)  # 进行评估
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0  # 计算标准化得分
            wandb.log({
                "epoch": epoch,
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })
    
    wandb.finish()  # 完成 WandB 记录


if __name__ == "__main__":
    main()
