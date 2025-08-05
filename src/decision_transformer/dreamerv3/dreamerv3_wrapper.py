"""
DreamerV3 Wrapper for Decision Transformer integration (stable single-device version)
"""

import sys
import os
import numpy as np
import ruamel.yaml as yaml
import random
import traceback
import jax
import jax.numpy as jnp
from jax import config as jax_config
import io

# --- DreamerV3 repo path setup ---
sys.path.append('/home/hail/Project/dreamerv3_jax')
dreamer_folder = os.path.dirname('/home/hail/Project/dreamerv3_jax/dreamerv3')
sys.path.insert(0, str(dreamer_folder))

import embodied
import embodied.jax
import elements
from dreamerv3 import agent as dreamer_agent

# Import PERBuffer from the separate module (adjust the relative path if needed)
from .per_buffer import PERBuffer


class DreamerV3Wrapper:
    """Wrapper for DreamerV3 (world model training) with safe single-device settings."""

    def __init__(self, obs_space, act_space, config_path=None):

        self.jax = jax
        self.jnp = jnp

        # ---- Load config and initialize agent ----
        if config_path is None:
            config_path = '/home/hail/Project/dreamerv3_jax/dreamerv3/configs.yaml'

        config = self._load_config(config_path)

        # Create agent (suppress noisy stdout during construction)
       
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.agent = dreamer_agent.Agent(obs_space, act_space, config=config)
        finally:
            sys.stdout = old_stdout
        self.replay_buffer = PERBuffer()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_config(self, config_path):
        """Load DreamerV3 config using elements.Config similar to main.py
        """
        try:
            with open(config_path, 'r') as f:
                configs = yaml.YAML(typ='safe').load(f)

            full_config = elements.Config(configs['defaults'])

            agent_config = elements.Config({
                **full_config.agent,
                'logdir': full_config.get('logdir', '/tmp/dreamer'),
                'seed': full_config.get('seed', 0),
                'jax': full_config.jax,
                'batch_size': 4,          
                'batch_length': 16,         
                'replay_context': full_config.get('replay_context', 1),
                'report_length': full_config.get('report_length', 32),
                'replica': 0,
                'replicas': 1,
            })

            agent_config = agent_config.update({
                'jax.prealloc': False,          # Avoid grabbing all memory
                'jax.compute_dtype': 'float32',
                'jax.policy_devices': [0],
                'jax.train_devices': [0],
                'jax.mock_devices': 0,
                'jax.expect_devices': 1,
                'jax.enable_policy': False,
                'jax.coordinator_address': '',
                'jax.jit': True,
                'jax.debug': False,
            })

            return agent_config

        except Exception as e:
            print(f"⚠️ Failed to load config from {config_path}: {e}")
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Training (World Model / Dynamics)
    # ------------------------------------------------------------------
    def dynamics_learning(self, batch_data):
        """Run a single DreamerV3 world model (dynamics) train step."""
        # Shapes from config
        batch_size = int(getattr(self.agent.config, 'batch_size', 16))
        batch_length = int(getattr(self.agent.config, 'batch_length', 64))
        replay_context = int(getattr(self.agent.config, 'replay_context', 1))
        seq_len = batch_length + replay_context

        # 1) Format -> 2) Adjust -> 3) Convert to JAX on single device
        formatted_np = self._format_batch_for_dreamer(batch_data)
        adjusted_np = self._adjust_batch_size(formatted_np, batch_size, seq_len)

        # Ensure seed exists
        if 'seed' not in adjusted_np:
            adjusted_np['seed'] = np.array(
                [random.randint(0, np.iinfo(np.uint32).max),
                 random.randint(0, np.iinfo(np.uint32).max)],
                dtype=np.uint32
            )

        # Convert to JAX arrays
        device = self.jax.devices()[0]
        to_device = lambda x: self.jax.device_put(self.jnp.asarray(x), device) \
                              if hasattr(x, 'shape') else x
        jax_data = {k: to_device(v) for k, v in adjusted_np.items()}

        # 4) Train
        carry = self.agent.init_train(batch_size=batch_size)
        carry, outputs, metrics = self.agent.train(carry, jax_data)
        return carry, outputs, metrics

    # ------------------------------------------------------------------
    # Behavior (stub)
    # ------------------------------------------------------------------
    def behavior_learning(self, carry, batch_data=None, *_, **__):
        """
        Behavior(정책/가치) 학습 단계.
        - train.py가 (carry, current_batch) 형태로 호출해도 받도록 batch_data, *_, **__ 허용
        - DreamerV3 Agent에 train_behavior가 있으면 사용, 없으면 no-op으로 안전 종료
        - 반환 형태는 (carry, outputs, metrics)로 train.py 기대에 맞춤
        """
        try:
            # 일부 구현체는 world model과 behavior를 분리해 train_behavior를 제공합니다.
            if hasattr(self.agent, 'train_behavior'):
                # 필요 시 batch_data를 전달하도록 인터페이스 확인 후 넘겨도 됨.
                # 여기서는 보수적으로 인자 없이 호출하고, 실패 시 예외 처리합니다.
                try:
                    return self.agent.train_behavior(carry)
                except TypeError:
                    # 시그니처가 (carry, batch)인 구현일 수도 있으니 재시도
                    return self.agent.train_behavior(carry, batch_data)
            else:
                # 별도 behavior 학습이 없다면 no-op
                outputs, metrics = {}, {}
                return carry, outputs, metrics
        except Exception as e:
            print(f"⚠️ DreamerV3 behavior learning skipped due to error: {e}")
            outputs, metrics = {}, {}
            return carry, outputs, metrics


    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------
    def _format_batch_for_dreamer(self, batch_data):
        """Fill a batch dict matching self.agent.spaces (shapes/dtypes).
        Unknown keys are ignored; missing expected keys are zero-filled.
        """
        try:
            # Derive B, T from provided arrays if possible
            B, T = None, None
            for k in ('observation', 'action', 'reward', 'states', 'actions', 'rewards'):
                v = batch_data.get(k) if isinstance(batch_data, dict) else None
                if v is not None and hasattr(v, 'shape') and v.ndim >= 2:
                    B, T = int(v.shape[0]), int(v.shape[1])
                    break

            if B is None or T is None:
                B = int(getattr(self.agent.config, 'batch_size', 16))
                T = int(getattr(self.agent.config, 'batch_length', 64)) + int(
                    getattr(self.agent.config, 'replay_context', 1)
                )

            out = {}

            # Helper: make observation (B,T,64,64,3) uint8
            def ensure_obs(arr):
                arr = np.asarray(arr)
                # Normalize to (B,T,H,W,C)
                if arr.ndim == 3:      # (B,H,W) -> add T=1,C
                    arr = arr[:, None, :, :, None]
                elif arr.ndim == 4:    # (B,H,W,C) -> add T=1
                    arr = arr[:, None, :, :, :]
                # else assume (B,T,H,W,C)

                B0, T0, h, w, c = arr.shape
                # Center place into 64x64
                if (h, w) != (64, 64):
                    canvas = np.zeros((B0, T0, 64, 64, c), dtype=arr.dtype)
                    sh = max(0, (64 - h) // 2)
                    sw = max(0, (64 - w) // 2)
                    ah, aw = min(h, 64), min(w, 64)
                    canvas[:, :, sh:sh+ah, sw:sw+aw, :] = arr[:, :, :ah, :aw, :]
                    arr = canvas

                # Force C=3
                if c != 3:
                    if c > 3:
                        arr = arr[..., :3]
                    else:
                        rep = (3 + c - 1) // c
                        arr = np.tile(arr, (1, 1, 1, 1, rep))[..., :3]

                # Dreamer uses uint8 images
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8, copy=False)
                return arr

            # If user passed 'states'/'actions'/'rewards', map them to Dreamer keys
            if 'states' in batch_data and batch_data['states'] is not None:
                out['observation'] = ensure_obs(
                    batch_data['states'].cpu().numpy() if hasattr(batch_data['states'], 'cpu')
                    else batch_data['states']
                )
            if 'actions' in batch_data and batch_data['actions'] is not None:
                a = batch_data['actions'].cpu().numpy() if hasattr(batch_data['actions'], 'cpu') \
                    else batch_data['actions']
                out['action'] = np.asarray(a, dtype=np.int32)
            if 'rewards' in batch_data and batch_data['rewards'] is not None:
                r = batch_data['rewards'].cpu().numpy() if hasattr(batch_data['rewards'], 'cpu') \
                    else batch_data['rewards']
                r = np.asarray(r)
                if r.ndim == 3 and r.shape[-1] == 1:
                    r = r.squeeze(-1)
                if r.ndim == 1:
                    r = r[:, None]
                out['reward'] = r.astype(np.float32)

            # If not provided, create zero rewards with proper (B,T)
            if 'reward' not in out:
                out['reward'] = np.zeros((B, T), dtype=np.float32)

            # Make sure observation exists; if not, create blank frames
            if 'observation' not in out:
                out['observation'] = np.zeros((B, T, 64, 64, 3), dtype=np.uint8)

            # Dreamer training signals
            out['is_first'] = np.zeros((B, T), dtype=bool)
            out['is_last'] = np.zeros((B, T), dtype=bool)
            out['is_terminal'] = np.zeros((B, T), dtype=bool)
            out['is_first'][:, 0] = True  # first time step

            # Seed (2 x uint32)
            seed = np.array(
                [random.randint(0, np.iinfo(np.uint32).max),
                 random.randint(0, np.iinfo(np.uint32).max)],
                dtype=np.uint32
            )
            out['seed'] = seed

            # Additionally, fill any other expected keys with zeros of the right shape/dtype
            # by consulting self.agent.spaces
            for key, space in self.agent.spaces.items():
                if key in out:
                    continue
                step_shape = tuple(int(d) for d in getattr(space, 'shape', ()))
                dtype = getattr(space, 'dtype', np.float32)
                # Most spaces are per-step; expand to (B,T,*step_shape)
                arr = np.zeros((B, T) + step_shape, dtype=dtype)
                # Type-specific fixes
                if key == 'action':
                    arr = arr.astype(np.int32, copy=False)
                elif key in ('is_first', 'is_last', 'is_terminal'):
                    arr = arr.astype(np.bool_, copy=False)
                elif key == 'observation':
                    arr = np.zeros((B, T, 64, 64, 3), dtype=np.uint8)
                out[key] = arr

            return out

        except Exception as e:
            print(f"⚠️ Error formatting batch for DreamerV3: {e}")
            traceback.print_exc()
            # Minimal fallback
            B = int(getattr(self.agent.config, 'batch_size', 16))
            T = int(getattr(self.agent.config, 'batch_length', 64)) + int(
                getattr(self.agent.config, 'replay_context', 1)
            )
            seed = np.array(
                [random.randint(0, np.iinfo(np.uint32).max),
                 random.randint(0, np.iinfo(np.uint32).max)],
                dtype=np.uint32
            )
            return {
                'observation': np.zeros((B, T, 64, 64, 3), dtype=np.uint8),
                'action': np.zeros((B, T), dtype=np.int32),
                'reward': np.zeros((B, T), dtype=np.float32),
                'is_first': np.pad(np.zeros((B, T-1), dtype=bool), ((0,0),(1,0)), constant_values=True),
                'is_last': np.zeros((B, T), dtype=bool),
                'is_terminal': np.zeros((B, T), dtype=bool),
                'seed': seed,
            }

    def _adjust_batch_size(self, data, target_batch_size, seq_len):
        """Adjust batch size and time length to (target_batch_size, seq_len).
        Also enforces (64,64,3) for observation and proper dtypes.
        """
        out = {}
        for key, value in data.items():
            if key == 'seed':
                # Keep as (2,) uint32
                out[key] = np.asarray(value, dtype=np.uint32)
                continue

            if not hasattr(value, 'shape'):
                out[key] = value
                continue

            arr = np.asarray(value)

            # 1) Adjust batch size
            B = arr.shape[0]
            if B >= target_batch_size:
                arr = arr[:target_batch_size]
            else:
                rep = (target_batch_size + B - 1) // B
                arr = np.tile(arr, (rep,) + (1,) * (arr.ndim - 1))[:target_batch_size]

            # 2) Adjust time length if temporal (ndim >= 2)
            if arr.ndim >= 2:
                T = arr.shape[1]
                if T < seq_len:
                    pad_len = seq_len - T
                    if key == 'observation':
                        last = arr[:, -1:, ...]
                        pad = np.tile(last, (1, pad_len) + (1,) * (last.ndim - 2))
                    elif key == 'action':
                        last = arr[:, -1:, ...]
                        pad = np.tile(last, (1, pad_len) + (1,) * (last.ndim - 2))
                    elif key == 'reward':
                        pad = np.zeros((arr.shape[0], pad_len) + arr.shape[2:], dtype=arr.dtype)
                    else:
                        pad = np.zeros((arr.shape[0], pad_len) + arr.shape[2:], dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=1)
                elif T > seq_len:
                    arr = arr[:, :seq_len]

            # 3) Observation fix to (64,64,3) uint8
            if key == 'observation' and arr.ndim == 5:
                b, t, h, w, c = arr.shape
                if (h, w) != (64, 64) or c != 3:
                    canvas = np.zeros((b, t, 64, 64, 3), dtype=arr.dtype)
                    # place resized/centered content
                    ah, aw = min(h, 64), min(w, 64)
                    sh = (64 - ah) // 2
                    sw = (64 - aw) // 2
                    if c >= 3:
                        src = arr[..., :3]
                    else:
                        rep = (3 + c - 1) // c
                        src = np.tile(arr, (1, 1, 1, 1, rep))[..., :3]
                    canvas[:, :, sh:sh+ah, sw:sw+aw, :] = src[:, :, :ah, :aw, :]
                    arr = canvas
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8, copy=False)

            # 4) Dtypes for common keys
            if key == 'action':
                arr = arr.astype(np.int32, copy=False)
            elif key == 'reward':
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
                arr = arr.astype(np.float32, copy=False)
            elif key in ('is_first', 'is_last', 'is_terminal'):
                arr = arr.astype(np.bool_, copy=False)

            out[key] = arr

        # Make sure the first step is marked as first
        if 'is_first' in out and out['is_first'].shape[1] > 0:
            out['is_first'][:, 0] = True

        return out
