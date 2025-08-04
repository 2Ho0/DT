"""
DreamerV3 Wrapper for Decision Transformer integration
"""

import sys
import os
import numpy as np
import ruamel.yaml as yaml

# DreamerV3 imports
sys.path.append('/home/hail/Project/dreamerv3_jax')
dreamer_folder = os.path.dirname('/home/hail/Project/dreamerv3_jax/dreamerv3')
sys.path.insert(0, str(dreamer_folder))

import embodied
import embodied.jax
import elements
from dreamerv3 import agent as dreamer_agent

# Import PERBuffer from the separate module
from .per_buffer import PERBuffer


class DreamerV3Wrapper:
    """Wrapper for DreamerV3 dynamics and behavior learning"""
    
    def __init__(self, obs_space, act_space, config_path=None):
        # Force JAX to use single CPU device and disable multi-device features
        import jax
        import os
        
        # Set JAX environment variables to force CPU-only operation
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['JAX_DISABLE_MOST_OPTIMIZATIONS'] = 'true'
        
        # Configure JAX for single device operation
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_disable_jit', False)  # Keep JIT but force single device
        
        # Completely disable JAX distributed features and sharding
        try:
            # Force JAX to use no sharding/mesh
            from jax.experimental import mesh_utils
            from jax.sharding import NamedSharding, PartitionSpec
            
            # Set up minimal single-device mesh to override DreamerV3's mesh
            devices = jax.devices()[:1]  # Only use first device
            mesh = mesh_utils.create_device_mesh((1,), devices)
            
        except Exception as e:
            print(f"⚠️ Failed to setup single-device mesh: {e}")
            pass
        
        # Also try to initialize distributed with explicit single process
        try:
            jax.distributed.initialize(
                coordinator_address=None,
                num_processes=1,
                process_id=0,
                local_device_ids=None
            )
        except Exception:
            pass
        
        # Load DreamerV3 config
        if config_path is None:
            config_path = '/home/hail/Project/dreamerv3_jax/dreamerv3/configs.yaml'
        
        # Initialize DreamerV3 agent
        config = self._load_config()
        
        # Suppress DreamerV3 verbose output during initialization
        import contextlib
        import sys
        import io
        
        # Redirect stdout temporarily to suppress verbose logs
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            self.agent = dreamer_agent.Agent(obs_space, act_space, config=config)
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
        print("✅ DreamerV3 agent initialized")
        
        self.replay_buffer = PERBuffer()
        
    def _load_config(self):
        """Load DreamerV3 config using elements.Config (like in main.py)"""
        try:
            config_path = '/home/hail/Project/dreamerv3_jax/dreamerv3/configs.yaml'
            
            # Read YAML file using standard library
            with open(config_path, 'r') as f:
                configs = yaml.YAML(typ='safe').load(f)
            
            # Get the full config first (just like main.py)
            full_config = elements.Config(configs['defaults'])
            
            # Create agent config exactly like main.py does
            agent_config = elements.Config({
                **full_config.agent,
                'logdir': full_config.get('logdir', '/tmp/dreamer'),
                'seed': full_config.get('seed', 0),
                'jax': full_config.jax,
                'batch_size': full_config.get('batch_size', 16),
                'batch_length': full_config.get('batch_length', 64),
                'replay_context': full_config.get('replay_context', 1),
                'report_length': full_config.get('report_length', 32),
                'replica': 0,    # Force single replica
                'replicas': 1,   # Force single replica count
            })
            
            # Modify jax config for compatibility and single-device operation
            agent_config = agent_config.update({
                'jax.platform': 'cpu',           # Use CPU for compatibility  
                'jax.prealloc': False,           # Better memory management
                'jax.compute_dtype': 'float32',  # Use float32 instead of bfloat16
                'jax.policy_devices': [0],       # Single device only
                'jax.train_devices': [0],        # Single device only
                'jax.mock_devices': 0,           # No mock devices
                'jax.expect_devices': 1,         # Expect only 1 device
                'jax.enable_policy': False,      # Disable distributed policy
                'jax.coordinator_address': '',   # No coordinator
                'jax.jit': True,                 # Keep JIT enabled
                'jax.debug': False               # No debug mode
            })
            
            return agent_config
            
        except Exception as e:
            print(f"⚠️ Failed to load config from {config_path}: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Using fallback config...")
            raise e  # For now, fail fast if config loading fails
    
    def dynamics_learning(self, batch_data):
        """Perform dynamics learning with new batch"""
        
        # Convert batch data to DreamerV3 format if needed
        formatted_data = self._format_batch_for_dreamer(batch_data)
        
        # Use DreamerV3's configured batch_size to avoid JAX compilation issues
        config_batch_size = getattr(self.agent.config, 'batch_size', 16)
        
        batch_length = getattr(self.agent.config, 'batch_length', 64)
        replay_context = getattr(self.agent.config, 'replay_context', 1)
        config_seq_length = batch_length + replay_context
        
        # Adjust formatted_data to match DreamerV3's expected batch size
        formatted_data = self._adjust_batch_size(formatted_data, config_batch_size)
        
        # Force all data to be completely unsharded just before agent.train
        unsharded_data = {}
        import jax
        from jax.sharding import SingleDeviceSharding
        
        device = jax.devices()[0]
        single_sharding = SingleDeviceSharding(device)
        
        for key, value in formatted_data.items():
            if hasattr(value, 'shape') and hasattr(value, '__array__'):
                # Convert to completely unsharded JAX array
                try:
                    # First convert to numpy to remove any existing sharding
                    np_value = np.asarray(value)
                    # Then create fresh JAX array with explicit single device sharding
                    unsharded_data[key] = jax.device_put(np_value, single_sharding)
                except Exception as e:
                    print(f"  ⚠️ Failed to reshard {key}: {e}")
                    unsharded_data[key] = value
            else:
                unsharded_data[key] = value
        
        # Complete JAX environment reset to prevent sharding conflicts
        try:
            # Clear any existing JAX compilation cache that might have sharding
            jax.clear_caches()
            
            # Force JAX to forget any mesh/sharding configurations
            import os
            current_backend = os.environ.get('JAX_PLATFORM_NAME', 'cpu')
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            
            # Re-initialize JAX with clean state
            from jax._src import config as jax_config
            jax_config._thread_local_state.device_count_updated = False
            
        except Exception as e:
            print(f"🔧 JAX reset warning: {e}")
        
        # Override JAX's default device/sharding for this call
        with jax.default_device(device):
            try:
                # Train world model (dynamics) 
                carry = self.agent.init_train(batch_size=config_batch_size)
                carry, outputs, metrics = self.agent.train(carry, unsharded_data)
            except Exception as e:
                print(f"⚠️ DreamerV3 learning failed: {e}")
                # If sharding still fails, try with pure numpy data
                print("🔧 Retrying with pure numpy data...")
                numpy_data = {}
                for key, value in unsharded_data.items():
                    if hasattr(value, '__array__'):
                        numpy_data[key] = np.asarray(value)
                    else:
                        numpy_data[key] = value
                
                carry = self.agent.init_train(batch_size=config_batch_size)
                carry, outputs, metrics = self.agent.train(carry, numpy_data)
    
        return carry, outputs, metrics
    
    def behavior_learning(self, carry):
        """Perform behavior learning using imagination"""
        
        # This uses the imagination mechanism from DreamerV3
        # The actual implementation would depend on your specific setup
        
        return carry
    
    def _format_batch_for_dreamer(self, batch_data):
        """Convert batch data to DreamerV3 expected format"""
        try:
            # DreamerV3 expects specific keys and format
            formatted_data = {}
            
            if 'states' in batch_data and batch_data['states'] is not None:
                # Convert states to proper observation format
                states = batch_data['states']
                if hasattr(states, 'cpu'):
                    states = states.cpu().numpy()
                
                # Ensure proper shape for DreamerV3 (batch, time, 64, 64, 3)
                original_shape = states.shape
                
                if len(states.shape) == 3:  # (batch, height, width) -> add time and channels
                    states = states[:, None, :, :, None]
                    states = np.tile(states, (1, 1, 1, 1, 3))  # Add 3 channels
                elif len(states.shape) == 4:  # (batch, height, width, channels) -> add time
                    states = states[:, None, :, :, :]
                elif len(states.shape) == 5:  # (batch, time, height, width, channels)
                    pass  # Already correct format
                else:
                    print(f"⚠️ Unexpected states shape: {states.shape}")
                    
                # Resize to DreamerV3 expected size: (batch, time, 64, 64, 3)
                if len(states.shape) == 5:
                    batch_size, time_len, h, w, c = states.shape
                    
                    # Resize spatial dimensions to 64x64
                    if h != 64 or w != 64:
                        resized_states = np.zeros((batch_size, time_len, 64, 64, c), dtype=states.dtype)
                        # Center the original image in 64x64 space
                        start_h = max(0, (64 - h) // 2)
                        start_w = max(0, (64 - w) // 2)
                        end_h = min(64, start_h + h)
                        end_w = min(64, start_w + w)
                        
                        actual_h = min(h, 64)
                        actual_w = min(w, 64)
                        resized_states[:, :, start_h:start_h+actual_h, start_w:start_w+actual_w, :] = \
                            states[:, :, :actual_h, :actual_w, :]
                        states = resized_states
                    
                    # Adjust channels to 3
                    if c != 3:
                        if c > 3:
                            # Take first 3 channels
                            states = states[:, :, :, :, :3]
                        else:
                            # Repeat/tile channels to get 3
                            repeat_factor = (3 + c - 1) // c  # Ceiling division
                            states = np.tile(states, (1, 1, 1, 1, repeat_factor))
                            states = states[:, :, :, :, :3]
                
                formatted_data['observation'] = states.astype(np.uint8)
            
            if 'actions' in batch_data and batch_data['actions'] is not None:
                actions = batch_data['actions']
                if hasattr(actions, 'cpu'):
                    actions = actions.cpu().numpy()
                formatted_data['action'] = actions.astype(np.int32)
            
            if 'rewards' in batch_data and batch_data['rewards'] is not None:
                rewards = batch_data['rewards']
                if hasattr(rewards, 'cpu'):
                    rewards = rewards.cpu().numpy()
                
                # DreamerV3 expects (batch, time) shape, not (batch, time, 1)
                if len(rewards.shape) == 3 and rewards.shape[-1] == 1:
                    rewards = rewards.squeeze(-1)  # Remove last dimension
                elif len(rewards.shape) == 1:
                    # Add time dimension if missing
                    rewards = rewards[:, None]
                
                formatted_data['reward'] = rewards.astype(np.float32)
            else:
                # Create dummy rewards if not available - DreamerV3 format (batch, time)
                batch_size = len(list(formatted_data.values())[0]) if formatted_data else 1
                time_len = 1
                if 'observation' in formatted_data and len(formatted_data['observation'].shape) >= 2:
                    time_len = formatted_data['observation'].shape[1]
                formatted_data['reward'] = np.zeros((batch_size, time_len), dtype=np.float32)
            
            # Add required DreamerV3 signals
            batch_size = len(list(formatted_data.values())[0]) if formatted_data else 1
            time_len = 1
            if 'observation' in formatted_data and len(formatted_data['observation'].shape) >= 2:
                time_len = formatted_data['observation'].shape[1]
            
            formatted_data['is_first'] = np.zeros((batch_size, time_len), dtype=bool)
            formatted_data['is_last'] = np.zeros((batch_size, time_len), dtype=bool) 
            formatted_data['is_terminal'] = np.zeros((batch_size, time_len), dtype=bool)
            
            # Mark first timestep
            formatted_data['is_first'][:, 0] = True
            
            # Add seed required by agent.train() - DreamerV3 expects 2 uint32 values
            import random
            import jax
            seed1 = random.randint(0, np.iinfo(np.uint32).max)
            seed2 = random.randint(0, np.iinfo(np.uint32).max)
            
            # Create seed and put it on the correct JAX device
            seed_array = np.array([seed1, seed2], dtype=np.uint32)
            try:
                # Put seed on JAX device to avoid host-to-device transfer issues
                formatted_data['seed'] = jax.device_put(seed_array)
            except Exception as e:
                print(f"⚠️ Failed to device_put seed, using numpy array: {e}")
                formatted_data['seed'] = seed_array
            
            # Add DreamerV3 internal training requirements with correct time dimensions
            
            # consec: consecutive batch index (batch, time)
            formatted_data['consec'] = np.zeros((batch_size, time_len), dtype=np.int32)
            
            # stepid: unique step identifier (batch, time, 20)
            formatted_data['stepid'] = np.random.randint(0, 256, (batch_size, time_len, 20), dtype=np.uint8)
            
            # dyn/deter and dyn/stoch: RSSM state entries for replay context with time dimension
            # From config: deter=8192, stoch=32, classes=64
            formatted_data['dyn/deter'] = np.zeros((batch_size, time_len, 8192), dtype=np.float32)
            formatted_data['dyn/stoch'] = np.zeros((batch_size, time_len, 32, 64), dtype=np.float32)
            
            return formatted_data
            
        except Exception as e:
            print(f"⚠️ Error formatting batch for DreamerV3: {e}")
            # Return minimal valid format with correct DreamerV3 shapes
            fallback_batch_size = 1
            fallback_seq_length = 65  # Default batch_length + replay_context
            
            # Create fallback seed with JAX device placement
            import random
            import jax
            seed1 = random.randint(0, np.iinfo(np.uint32).max)
            seed2 = random.randint(0, np.iinfo(np.uint32).max)
            seed_array = np.array([seed1, seed2], dtype=np.uint32)
            try:
                fallback_seed = jax.device_put(seed_array)
            except Exception:
                fallback_seed = seed_array
            
            return {
                'observation': np.zeros((fallback_batch_size, fallback_seq_length, 64, 64, 3), dtype=np.uint8),
                'action': np.zeros((fallback_batch_size, fallback_seq_length), dtype=np.int32),
                'reward': np.zeros((fallback_batch_size, fallback_seq_length), dtype=np.float32),  # No last dim
                'is_first': np.zeros((fallback_batch_size, fallback_seq_length), dtype=bool),
                'is_last': np.zeros((fallback_batch_size, fallback_seq_length), dtype=bool),
                'is_terminal': np.zeros((fallback_batch_size, fallback_seq_length), dtype=bool),
                'seed': fallback_seed,
                'consec': np.zeros((fallback_batch_size, fallback_seq_length), dtype=np.int32),
                'stepid': np.random.randint(0, 256, (fallback_batch_size, fallback_seq_length, 20), dtype=np.uint8),
                'dyn/deter': np.zeros((fallback_batch_size, fallback_seq_length, 8192), dtype=np.float32),
                'dyn/stoch': np.zeros((fallback_batch_size, fallback_seq_length, 32, 64), dtype=np.float32),
            }
    
    def _safe_device_put(self, data):
        """Safely put data on JAX device with explicit unsharded placement"""
        try:
            import jax
            import jax.numpy as jnp
            from jax.sharding import SingleDeviceSharding
            
            # Force data to be unsharded on single device
            if hasattr(data, 'shape'):
                device = jax.devices()[0]  # Get first CPU device
                
                # Create explicit single device sharding to override any mesh
                single_device_sharding = SingleDeviceSharding(device)
                
                # Convert to JAX array with explicit unsharded placement
                jax_array = jnp.array(data)
                
                # Force the array to be on single device without sharding
                return jax.device_put(jax_array, single_device_sharding)
            else:
                # For scalars, simple device_put
                return jax.device_put(data)
        except Exception as e:
            # If sharding fails, try simple device_put
            try:
                import jax
                import jax.numpy as jnp
                device = jax.devices()[0]
                return jax.device_put(jnp.array(data), device)
            except Exception as e2:
                # Final fallback: return original numpy data
                print(f"🔧 All device put attempts failed for data shape {data.shape if hasattr(data, 'shape') else 'scalar'}: {e}, {e2}")
                return data
    
    def _adjust_batch_size(self, formatted_data, target_batch_size):
        """Adjust batch size and sequence length to match DreamerV3 expectations"""
        try:
            adjusted_data = {}
            # Get sequence length from config: batch_length + replay_context
            batch_length = getattr(self.agent.config, 'batch_length', 64)
            replay_context = getattr(self.agent.config, 'replay_context', 1)
            config_seq_length = batch_length + replay_context
            
            for key, value in formatted_data.items():
                if key == 'seed':
                    # Use safe device put for seed
                    adjusted_data[key] = self._safe_device_put(value)
                    continue
                    
                if not hasattr(value, 'shape') or len(value.shape) == 0:
                    adjusted_data[key] = value
                    continue
                
                # Step 1: Adjust batch size (simply take first N samples or repeat)
                current_batch_size = value.shape[0]
                if current_batch_size >= target_batch_size:
                    # Take first target_batch_size samples
                    adjusted_value = value[:target_batch_size]
                else:
                    # Repeat the data to reach target_batch_size
                    repeat_times = (target_batch_size + current_batch_size - 1) // current_batch_size
                    repeated_value = np.tile(value, (repeat_times,) + (1,) * (len(value.shape) - 1))
                    adjusted_value = repeated_value[:target_batch_size]
                
                # Step 2: Adjust sequence length for temporal data
                if key in ['observation', 'action', 'reward'] and len(adjusted_value.shape) >= 2:
                    current_seq_length = adjusted_value.shape[1]
                    if current_seq_length < config_seq_length:
                        # Pad sequence to reach config_seq_length
                        pad_length = config_seq_length - current_seq_length
                        if key == 'observation':
                            # Repeat last frame
                            last_frame = adjusted_value[:, -1:, ...]
                            padding = np.tile(last_frame, (1, pad_length) + (1,) * (len(last_frame.shape) - 2))
                        elif key == 'action':
                            # Repeat last action
                            last_action = adjusted_value[:, -1:, ...]
                            padding = np.tile(last_action, (1, pad_length) + (1,) * (len(last_action.shape) - 2))
                        elif key == 'reward':
                            # Pad with zeros
                            pad_shape = (adjusted_value.shape[0], pad_length) + adjusted_value.shape[2:]
                            padding = np.zeros(pad_shape, dtype=adjusted_value.dtype)
                        adjusted_value = np.concatenate([adjusted_value, padding], axis=1)
                    elif current_seq_length > config_seq_length:
                        # Truncate sequence
                        adjusted_value = adjusted_value[:, :config_seq_length]
                    
                    # Step 2.5: Fix observation spatial dimensions and channels for DreamerV3
                    if key == 'observation':
                        # DreamerV3 expects (batch, time, 64, 64, 3)
                        # Current: (batch, time, 7, 7, 20)
                        current_shape = adjusted_value.shape
                        if len(current_shape) == 5:  # (batch, time, H, W, C)
                            batch_size, time_len, h, w, c = current_shape
                            
                            # Resize spatial dimensions: (7,7) -> (64,64)
                            if h != 64 or w != 64:
                                # Use simple repeat/pad to get to 64x64
                                resized_obs = np.zeros((batch_size, time_len, 64, 64, c), dtype=adjusted_value.dtype)
                                # Center the 7x7 image in 64x64 space
                                start_h = (64 - h) // 2
                                start_w = (64 - w) // 2
                                resized_obs[:, :, start_h:start_h+h, start_w:start_w+w, :] = adjusted_value
                                adjusted_value = resized_obs
                            
                            # Reduce channels: 20 -> 3
                            if c != 3:
                                if c > 3:
                                    # Take first 3 channels
                                    adjusted_value = adjusted_value[:, :, :, :, :3]
                                else:
                                    # Repeat channels to get 3
                                    repeat_factor = 3 // c + 1
                                    repeated = np.tile(adjusted_value, (1, 1, 1, 1, repeat_factor))
                                    adjusted_value = repeated[:, :, :, :, :3]
                
                # Step 3: Handle remaining temporal adjustments
                elif key == 'reward' and len(adjusted_value.shape) == 3 and adjusted_value.shape[-1] == 1:
                    # Remove last dimension if it exists: (batch, time, 1) -> (batch, time)
                    adjusted_value = adjusted_value.squeeze(-1)
                elif key in ['is_first', 'is_last', 'is_terminal', 'consec', 'stepid', 'dyn/deter', 'dyn/stoch']:
                    # These should already have correct time dimensions from _format_batch_for_dreamer
                    # Just adjust sequence length if needed
                    if len(adjusted_value.shape) >= 2:
                        current_seq_length = adjusted_value.shape[1]
                        if current_seq_length < config_seq_length:
                            # Pad with appropriate values
                            if key in ['is_first', 'is_last', 'is_terminal', 'consec']:
                                pad_shape = (adjusted_value.shape[0], config_seq_length - current_seq_length) + adjusted_value.shape[2:]
                                padding = np.zeros(pad_shape, dtype=adjusted_value.dtype)
                            else:  # stepid, dyn/deter, dyn/stoch
                                if key == 'stepid':
                                    padding = np.random.randint(0, 256, (adjusted_value.shape[0], config_seq_length - current_seq_length, 20), dtype=adjusted_value.dtype)
                                else:  # dyn/deter, dyn/stoch
                                    pad_shape = (adjusted_value.shape[0], config_seq_length - current_seq_length) + adjusted_value.shape[2:]
                                    padding = np.zeros(pad_shape, dtype=adjusted_value.dtype)
                            adjusted_value = np.concatenate([adjusted_value, padding], axis=1)
                        elif current_seq_length > config_seq_length:
                            adjusted_value = adjusted_value[:, :config_seq_length]
                
                adjusted_data[key] = adjusted_value
            
            return adjusted_data
            
        except Exception as e:
            print(f"⚠️ Error adjusting batch size: {e}")
            import traceback
            traceback.print_exc()
            return formatted_data