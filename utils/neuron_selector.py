import torch
import numpy as np
import os
import json
from tqdm import tqdm

class NeuronManager:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.config = model.config.get_text_config()
        self.layers = model.language_model.layers

        self.num_layers = self.config.num_hidden_layers
        self.intermediate_size = self.config.intermediate_size 
        self.hooks = []
        self._temp_buffer = {i: [] for i in range(self.num_layers)}
        self.captured_data = {
            "jailbreak": {i: [] for i in range(self.num_layers)},
            "malicious":   {i: [] for i in range(self.num_layers)}, 
            "benign":    {i: [] for i in range(self.num_layers)} 
        }

        self.backup_weights = {}  

    def apply_intervention(self, patch_dict, top_k=None, scale_factor=0.0):

        self.reset_weights()
        count = 0

        with torch.no_grad():
            for layer_str, indices in patch_dict.items():
                layer_idx = int(layer_str)
                
                target_indices = indices
                if top_k is not None:
                    target_indices = indices[:top_k]
                
                if not target_indices:
                    continue

                layer_module = self.layers[layer_idx]
                target_layer = getattr(layer_module.mlp, 'down_proj')
                if target_layer is None: continue

                if layer_idx not in self.backup_weights:
                    self.backup_weights[layer_idx] = {}
                
                idx_tensor = torch.tensor(target_indices, device=self.device)
                original_cols = target_layer.weight.data[:, idx_tensor].clone()
                
                self.backup_weights[layer_idx] = (idx_tensor, original_cols)
                
                target_layer.weight.data[:, idx_tensor] *= scale_factor
                
                count += len(target_indices)

        print(f"[Manager] Intervention complete. Modified {count} neurons.")

    def reset_weights(self):

        if not self.backup_weights:
            return

        with torch.no_grad():
            for layer_idx, (indices, original_vals) in self.backup_weights.items():
                
                layer_module = self.layers[layer_idx]
                target_layer = getattr(layer_module.mlp, 'down_proj')
                
                target_layer.weight.data[:, indices] = original_vals
        
        self.backup_weights.clear() 

    def load_patch_file(self, json_path, top_k=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Patch file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        raw_neurons = data['robust_neurons']['neurons']
        
        if isinstance(raw_neurons, list):
            if top_k is not None:
                raw_neurons = raw_neurons[:top_k]

            patch_dict = {}
            for item in raw_neurons:
                l = str(item['layer'])
                n = item['neuron']

                if l not in patch_dict:
                    patch_dict[l] = []
                patch_dict[l].append(n)
            return patch_dict
            
        elif isinstance(raw_neurons, dict):
            return raw_neurons
        else:
            raise ValueError("Unknown format in patch file.")

    def register_hooks(self, hook_target: str = "down"):
        self.clear_temp_buffer()
        self._hooks = []
        
        def get_gate_hook(layer_idx, act_fn):
            def hook(module, input, output):
                activated_val = act_fn(output)
                
                data = activated_val[:, -1, :].detach().cpu()
                self._temp_buffer[layer_idx].append(data)
            return hook

        def get_down_hook(layer_idx):
            def hook(module, input, output):
                data = input[0][:, -1, :].detach().cpu()
                self._temp_buffer[layer_idx].append(data)
            return hook

        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            if hook_target == "gate":
                target_module = None
                if hasattr(mlp, 'gate_proj'):
                    target_module = mlp.gate_proj
                
                if target_module:
                    act_fn = mlp.act_fn 
                    h = target_module.register_forward_hook(get_gate_hook(i, act_fn))
                    self._hooks.append(h)
                else:
                    raise ValueError(f"Warning: Layer {i} has no gate_proj.")
                
            elif hook_target == "down":
                target_module = None
                if hasattr(mlp, 'down_proj'):
                    target_module = mlp.down_proj
                
                if target_module:
                    h = target_module.register_forward_hook(get_down_hook(i))
                    self._hooks.append(h)
                else:
                    raise ValueError(f"Warning: Layer {i} has no down_proj.")
            else:
                raise ValueError(f"Unknown hook_target: {hook_target}. Use 'down' or 'gate'.")
        print("Hooks registered. Listening to activations...")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear_temp_buffer(self):
        self._temp_buffer = {i: [] for i in range(self.num_layers)}

    def flush_buffer_to_storage(self, label: str):
        if label not in self.captured_data:
            raise ValueError(f"Unknown label: {label}. Use 'jailbreak', 'malicious', or 'benign'.")
            
        print(f"Moving captured data to '{label}' storage...")
        for layer in range(self.num_layers):
            if len(self._temp_buffer[layer]) > 0:
                batch_data = torch.cat(self._temp_buffer[layer], dim=0).float().numpy()
                self.captured_data[label][layer].append(batch_data)
        
        self.clear_temp_buffer()


    def reset_captured_data(self):
        self.clear_temp_buffer()
        self.captured_data = {
            "jailbreak": {i: [] for i in range(self.num_layers)},
            "malicious":   {i: [] for i in range(self.num_layers)},
            "benign":    {i: [] for i in range(self.num_layers)}
        }



class Qwen2NeuronManager:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.config = model.config
        self.layers = model.model.language_model.layers

        self.num_layers = self.config.num_hidden_layers
        self.intermediate_size = self.config.intermediate_size
        self.hooks = []
        self._temp_buffer = {i: [] for i in range(self.num_layers)}
        self.captured_data = {
            "jailbreak": {i: [] for i in range(self.num_layers)},
            "malicious":   {i: [] for i in range(self.num_layers)},
            "benign":    {i: [] for i in range(self.num_layers)}  
        }

        self.backup_weights = {}  

    def apply_intervention(self, patch_dict, scale_factor=0.0):

        self.reset_weights()
        count = 0

        with torch.no_grad():
            for layer_str, indices in patch_dict.items():
                layer_idx = int(layer_str)
                
                target_indices = indices
                
                if not target_indices:
                    continue

                layer_module = self.layers[layer_idx]
                target_layer = getattr(layer_module.mlp, 'down_proj')
                if target_layer is None: continue

                if layer_idx not in self.backup_weights:
                    self.backup_weights[layer_idx] = {}
                
                idx_tensor = torch.tensor(target_indices, device=self.device)
                original_cols = target_layer.weight.data[:, idx_tensor].clone()
                
                self.backup_weights[layer_idx] = (idx_tensor, original_cols)
                
                target_layer.weight.data[:, idx_tensor] *= scale_factor
                
                count += len(target_indices)

        print(f"[Manager] Intervention complete. Modified {count} neurons.")

    def reset_weights(self):

        if not self.backup_weights:
            return

        with torch.no_grad():
            for layer_idx, (indices, original_vals) in self.backup_weights.items():
                
                layer_module = self.layers[layer_idx]
                target_layer = getattr(layer_module.mlp, 'down_proj')
                
                target_layer.weight.data[:, indices] = original_vals
        
        self.backup_weights.clear() 

    def load_patch_file(self, json_path, top_k=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Patch file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        raw_neurons = data['neurons']
        
        if isinstance(raw_neurons, list):
            if top_k is not None:
                raw_neurons = raw_neurons[:top_k]

            patch_dict = {}
            for item in raw_neurons:
                l = str(item['layer'])
                n = item['neuron']

                if l not in patch_dict:
                    patch_dict[l] = []
                patch_dict[l].append(n)
            return patch_dict
            
        elif isinstance(raw_neurons, dict):
            return raw_neurons
        else:
            raise ValueError("Unknown format in patch file.")

    def register_hooks(self, hook_target: str = "down"):
        self.clear_temp_buffer()
        self._hooks = []
        
        def get_gate_hook(layer_idx, act_fn):

            def hook(module, input, output):
                activated_val = act_fn(output)
                
                data = activated_val[:, -1, :].detach().cpu()
                self._temp_buffer[layer_idx].append(data)
            return hook

        def get_down_hook(layer_idx):
            def hook(module, input, output):
                data = input[0][:, -1, :].detach().cpu()
                self._temp_buffer[layer_idx].append(data)
            return hook

        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            if hook_target == "gate":
                target_module = None
                if hasattr(mlp, 'gate_proj'):
                    target_module = mlp.gate_proj
                
                if target_module:
                    act_fn = mlp.act_fn 
                    h = target_module.register_forward_hook(get_gate_hook(i, act_fn))
                    self._hooks.append(h)
                else:
                    raise ValueError(f"Warning: Layer {i} has no gate_proj.")
                
            elif hook_target == "down":
                target_module = None
                if hasattr(mlp, 'down_proj'):
                    target_module = mlp.down_proj
                
                if target_module:
                    h = target_module.register_forward_hook(get_down_hook(i))
                    self._hooks.append(h)
                else:
                    raise ValueError(f"Warning: Layer {i} has no down_proj.")
            else:
                raise ValueError(f"Unknown hook_target: {hook_target}. Use 'down' or 'gate'.")
        print("Hooks registered. Listening to activations...")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear_temp_buffer(self):
        self._temp_buffer = {i: [] for i in range(self.num_layers)}

    def flush_buffer_to_storage(self, label: str):

        if label not in self.captured_data:
            raise ValueError(f"Unknown label: {label}. Use 'jailbreak', 'malicious', or 'benign'.")
            
        print(f"Moving captured data to '{label}' storage...")
        for layer in range(self.num_layers):
            if len(self._temp_buffer[layer]) > 0:
                batch_data = torch.cat(self._temp_buffer[layer], dim=0).float().numpy()
                self.captured_data[label][layer].append(batch_data)
        
        self.clear_temp_buffer()

    def reset_captured_data(self):
        self.clear_temp_buffer()
        self.captured_data = {
            "jailbreak": {i: [] for i in range(self.num_layers)},
            "malicious":   {i: [] for i in range(self.num_layers)},
            "benign":    {i: [] for i in range(self.num_layers)}
        }


class HFNeuronManager:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.config = model.config.text_config
        self.layers = model.language_model.layers

        self.num_layers = self.config.num_hidden_layers
        self.intermediate_size = self.config.intermediate_size 
        self.hooks = []
        self._temp_buffer = {i: [] for i in range(self.num_layers)}
        self.captured_data = {
            "jailbreak": {i: [] for i in range(self.num_layers)},
            "malicious":   {i: [] for i in range(self.num_layers)},
            "benign":    {i: [] for i in range(self.num_layers)} 
        }

        self.backup_weights = {}  

    def apply_intervention(self, patch_dict, top_k=None, scale_factor=0.0):

        self.reset_weights()
        count = 0

        with torch.no_grad():
            for layer_str, indices in patch_dict.items():
                layer_idx = int(layer_str)
                
                target_indices = indices
                if top_k is not None:
                    target_indices = indices[:top_k]
                
                if not target_indices:
                    continue

                layer_module = self.layers[layer_idx]
                target_layer = getattr(layer_module.mlp, 'down_proj')
                if target_layer is None: continue

                if layer_idx not in self.backup_weights:
                    self.backup_weights[layer_idx] = {}
                
                idx_tensor = torch.tensor(target_indices, device=self.device)
                original_cols = target_layer.weight.data[:, idx_tensor].clone()
                
                self.backup_weights[layer_idx] = (idx_tensor, original_cols)
                
                target_layer.weight.data[:, idx_tensor] *= scale_factor
                
                count += len(target_indices)

        print(f"[Manager] Intervention complete. Modified {count} neurons.")

    def reset_weights(self):

        if not self.backup_weights:
            return

        with torch.no_grad():
            for layer_idx, (indices, original_vals) in self.backup_weights.items():
                
                layer_module = self.layers[layer_idx]
                target_layer = getattr(layer_module.mlp, 'down_proj')
                
                target_layer.weight.data[:, indices] = original_vals
        
        self.backup_weights.clear() 

    def load_patch_file(self, json_path, top_k=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Patch file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        raw_neurons = data['neurons']
        
        if isinstance(raw_neurons, list):
            if top_k is not None:
                raw_neurons = raw_neurons[:top_k]

            patch_dict = {}
            for item in raw_neurons:
                l = str(item['layer'])
                n = item['neuron']

                if l not in patch_dict:
                    patch_dict[l] = []
                patch_dict[l].append(n)
            return patch_dict
            
        elif isinstance(raw_neurons, dict):
            return raw_neurons
        else:
            raise ValueError("Unknown format in patch file.")

    def register_hooks(self, hook_target: str = "down"):
        self.clear_temp_buffer()
        self._hooks = []
        
        def get_gate_hook(layer_idx, act_fn):
            def hook(module, input, output):
                activated_val = act_fn(output)
                
                data = activated_val[:, -1, :].detach().cpu()
                self._temp_buffer[layer_idx].append(data)
            return hook

        def get_down_hook(layer_idx):
            def hook(module, input, output):
                data = input[0][:, -1, :].detach().cpu()
                self._temp_buffer[layer_idx].append(data)
            return hook

        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            if hook_target == "gate":
                target_module = None
                if hasattr(mlp, 'gate_proj'):
                    target_module = mlp.gate_proj
                
                if target_module:
                    act_fn = mlp.act_fn 
                    h = target_module.register_forward_hook(get_gate_hook(i, act_fn))
                    self._hooks.append(h)
                else:
                    raise ValueError(f"Warning: Layer {i} has no gate_proj.")
                
            elif hook_target == "down":
                target_module = None
                if hasattr(mlp, 'down_proj'):
                    target_module = mlp.down_proj
                
                if target_module:
                    h = target_module.register_forward_hook(get_down_hook(i))
                    self._hooks.append(h)
                else:
                    raise ValueError(f"Warning: Layer {i} has no down_proj.")
            else:
                raise ValueError(f"Unknown hook_target: {hook_target}. Use 'down' or 'gate'.")
        print("Hooks registered. Listening to activations...")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear_temp_buffer(self):
        self._temp_buffer = {i: [] for i in range(self.num_layers)}

    def flush_buffer_to_storage(self, label: str):

        if label not in self.captured_data:
            raise ValueError(f"Unknown label: {label}. Use 'jailbreak' or 'safe'.")
            
        print(f"Moving captured data to '{label}' storage...")
        for layer in range(self.num_layers):
            if len(self._temp_buffer[layer]) > 0:
                batch_data = torch.cat(self._temp_buffer[layer], dim=0).float().numpy()
                self.captured_data[label][layer].append(batch_data)
        
        self.clear_temp_buffer()


    def reset_captured_data(self):
        self.clear_temp_buffer()
        self.captured_data = {
            "jailbreak": {i: [] for i in range(self.num_layers)},
            "malicious":   {i: [] for i in range(self.num_layers)},
            "benign":    {i: [] for i in range(self.num_layers)}
        }
