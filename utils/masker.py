import torch

class AttentionMasker:
    """Attention head masker for analyzing safe attention heads in language models
    
    This class provides a mechanism to temporarily modify (mask) specific attention
    head weights in language models to analyze their impact on model safety outputs.
    
    Attributes:
        model: The language model to be analyzed
        head_mask: Configuration of heads to mask {(layer, head): [qkv types list]}
        mask_type: Type of masking, either "scale_mask" or "zero_mask"
        scale_factor: The scaling factor when using "scale_mask"
        hooks: List of model hooks
        original_weights: Dictionary to store original weights
        num_heads: Number of attention heads per layer
        head_dim: Dimension of each attention head
    """
    def __init__(self, model, head_mask=None, mask_type="scale_mask", scale_factor=1e-5, o_scale_factor=None):
        """Initialize the attention head masker
        
        Args:
            model: The language model to be analyzed
            head_mask: Configuration of heads to mask {(layer, head): [qkv types list]}
            mask_type: Type of masking, either "scale_mask" or "zero_mask"
            scale_factor: The scaling factor when using "scale_mask"
        """
        self.model = model
        self.head_mask = head_mask
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.o_scale_factor = o_scale_factor if o_scale_factor is not None else scale_factor
        self.hooks = []
        self.original_weights = {}
        self.extracted_features = {}
        
        # Get parameters from model configuration
        self.num_heads = getattr(model.config.get_text_config(), "num_attention_heads", 32)
        self.head_dim = getattr(model.config.get_text_config(), "hidden_size", 4096) // self.num_heads
        self.num_layers = getattr(model.config.get_text_config(), "num_hidden_layers")
    
    def apply_masking_hooks(self, clear_old=True):
        """Apply hooks to the attention layers of the language model
        
        This method adds pre-forward and post-forward hooks to each attention layer
        of the model to temporarily modify and restore attention head weights during
        forward propagation.
        """
        if self.head_mask is None:
            return
            
        if clear_old:
            self.remove_hooks()
        
        # Add hooks for language model
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            if hasattr(layer, 'self_attn'):
                pre_hook = layer.self_attn.register_forward_pre_hook(
                    lambda module, inputs, layer_idx=layer_idx: 
                        self._pre_attention_hook(module, inputs, layer_idx)
                )
                self.hooks.append(pre_hook)
                
                post_hook = layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, layer_idx=layer_idx:
                        self._post_attention_hook(module, inputs, output, layer_idx)
                )
                self.hooks.append(post_hook)

    def _pre_attention_hook(self, module, inputs, layer_idx):
        """Modify attention weights before forward propagation
        
        Modifies weights of specific attention heads according to head_mask
        configuration before model forward pass begins.
        
        Args:
            module: The current module being processed
            inputs: The module inputs
            layer_idx: The current layer index
        """
        if not hasattr(module, 'q_proj'):
            return
            
        # Process heads that need to be masked
        for (masked_layer, masked_head), qkv_types in self.head_mask.items():
            if masked_layer == layer_idx:
                key = f"layer_{layer_idx}_{masked_head}"
                
                # Process q/k/v projections
                for proj_type in qkv_types:
                    proj_name = f"{proj_type}_proj"
                    if hasattr(module, proj_name):
                        proj = getattr(module, proj_name)
                        start_idx = masked_head * self.head_dim
                        end_idx = (masked_head + 1) * self.head_dim
                        
                        if proj_type == 'o':
                            self.original_weights[f"{key}_{proj_type}"] = proj.weight[:, start_idx:end_idx].clone()
                            proj.weight.data[:, start_idx:end_idx] *= self.o_scale_factor
                        else:
                            # Save original weights
                            self.original_weights[f"{key}_{proj_type}"] = proj.weight[start_idx:end_idx].clone()

                            # Apply mask
                            if self.mask_type == "scale_mask":
                                proj.weight.data[start_idx:end_idx] *= self.scale_factor
                            elif self.mask_type == "zero_mask":
                                proj.weight.data[start_idx:end_idx] = 0
    
    def _post_attention_hook(self, module, inputs, output, layer_idx):
        """Restore attention weights after forward propagation
        
        Restores the modified attention head weights after module's forward
        propagation completes.
        
        Args:
            module: The current module being processed
            inputs: The module inputs
            output: The module outputs
            layer_idx: The current layer index
            
        Returns:
            The module outputs
        """
        if not hasattr(module, 'q_proj'):
            return output
            
        # Restore weights
        for key, weight in list(self.original_weights.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    if proj_type in ['q', 'k', 'v']:
                        proj.weight.data[start_idx:end_idx] = weight
                    elif proj_type == 'o':
                        proj.weight.data[:, start_idx:end_idx] = weight
                    del self.original_weights[key]
        
        return output
    
    def remove_hooks(self):
        """Remove all hooks and restore original weights
        
        Removes all hooks added to the model and ensures all modified weights
        are restored to their original state.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Ensure all original weights are restored
        for key, weight in list(self.original_weights.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.language_model.layers[layer_idx].self_attn
                start_idx = head_idx * self.head_dim
                end_idx = (head_idx + 1) * self.head_dim
                
                if proj_type == 'q' and hasattr(module, 'q_proj'):
                    module.q_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'k' and hasattr(module, 'k_proj'):
                    module.k_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'v' and hasattr(module, 'v_proj'):
                    module.v_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'o' and hasattr(module, 'o_proj'):
                    module.o_proj.weight.data[:, start_idx:end_idx] = weight
                
            except Exception as e:
                print(f"Error when restoring weights: {key}, Error: {str(e)}")
                
        self.original_weights = {}
    
    # Part 2: Feature Extraction
    def apply_extraction_hooks(self, extract_k=1, clear_old=True):

        if clear_old:
            self.remove_hooks()

        self.clear_extracted_data()

        for layer_idx, layer in enumerate(self.model.language_model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):

                hook = layer.self_attn.o_proj.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx:
                        self._extraction_hook(module, input, output, layer_idx, extract_k=extract_k)
                )
                self.hooks.append(hook)

    def _extraction_hook(self, module, input, output, layer_idx, extract_k):
        """
        Hook function taking extract_k as an argument.
        """
        # input[0] shape: [batch, seq_len, hidden_size]
        hidden_states = input[0].detach()
        bsz, seq_len, _ = hidden_states.size()
        
        # shape: [batch, k, hidden_size]
        last_k_tokens = hidden_states[:, -extract_k:, :]
        
        # target shape: [batch, k, num_heads, head_dim]
        act_reshaped = last_k_tokens.view(bsz, extract_k, self.num_heads, self.head_dim).cpu()
        
        # Store
        if layer_idx not in self.extracted_features:
            self.extracted_features[layer_idx] = []
        self.extracted_features[layer_idx].append(act_reshaped)

    def get_extracted_data(self, clear=True):

        if not self.extracted_features:
            print("Warning: No features extracted")
            return None
        
        sorted_layers = sorted(self.extracted_features.keys())
        all_layers_data = []

        for l in sorted_layers:
            layer_batches = self.extracted_features[l]
            layer_tensor = torch.cat(layer_batches, dim=0)
            
            all_layers_data.append(layer_tensor.unsqueeze(1))
            
        final_tensor = torch.cat(all_layers_data, dim=1)
        
        if clear:
            self.clear_extracted_data()
        
        return final_tensor

    def clear_extracted_data(self):
        self.extracted_features = {}


class Qwen2AttentionMasker:
    def __init__(self, model, head_mask=None, mask_type="scale_mask", scale_factor=1e-5, o_scale_factor=None):
        self.model = model
        self.head_mask = head_mask
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.o_scale_factor = o_scale_factor if o_scale_factor is not None else scale_factor
        self.hooks = []
        self.original_weights = {}
        self.original_biases = {}
        self.extracted_features = {}
        
        # Get parameters from model configuration
        self.num_heads = getattr(model.config, "num_attention_heads", 32)
        self.head_dim = getattr(model.config, "hidden_size", 4096) // self.num_heads
        self.num_layers = getattr(model.config, "num_hidden_layers")
    
    def apply_masking_hooks(self, clear_old=True):
        """Apply hooks to the attention layers of the language model.
        
        This method adds pre-forward and post-forward hooks to each attention layer
        of the model to temporarily modify and restore attention head weights during
        forward propagation.
        """
        if self.head_mask is None:
            return
        
        if clear_old:
            self.remove_hooks()
        
        # Add hooks for the language model
        for layer_idx, layer in enumerate(self.model.model.language_model.layers):
            if hasattr(layer, 'self_attn'):
                pre_hook = layer.self_attn.register_forward_pre_hook(
                    lambda module, inputs, layer_idx=layer_idx: 
                        self._pre_attention_hook(module, inputs, layer_idx)
                )
                self.hooks.append(pre_hook)

                print(f"Registered pre-hook for layer {layer_idx}")
                
                post_hook = layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, layer_idx=layer_idx:
                        self._post_attention_hook(module, inputs, output, layer_idx)
                )
                self.hooks.append(post_hook)

    def _pre_attention_hook(self, module, inputs, layer_idx):
        if not hasattr(module, 'q_proj'):
            return
            
        # Process heads that need to be masked
        for (masked_layer, masked_head), qkv_types in self.head_mask.items():
            if masked_layer == layer_idx:
                key = f"layer_{layer_idx}_{masked_head}"
                
                # Process q/k/v projections
                for proj_type in qkv_types:
                    proj_name = f"{proj_type}_proj"
                    if hasattr(module, proj_name):
                        proj = getattr(module, proj_name)
                        start_idx = masked_head * self.head_dim
                        end_idx = (masked_head + 1) * self.head_dim
                        
                        if proj_type == 'o':
                            self.original_weights[f"{key}_{proj_type}"] = proj.weight[:, start_idx:end_idx].clone()
                            proj.weight.data[:, start_idx:end_idx] *= self.o_scale_factor
                        else:
                            self.original_weights[f"{key}_{proj_type}"] = proj.weight[start_idx:end_idx].clone()
                            # Apply weight mask
                            if self.mask_type == "scale_mask":
                                proj.weight.data[start_idx:end_idx] *= self.scale_factor
                            elif self.mask_type == "zero_mask":
                                proj.weight.data[start_idx:end_idx] = 0
                        
                        # If there is a bias, process the bias as well
                        if hasattr(proj, 'bias') and proj.bias is not None:
                            # Save original bias
                            self.original_biases[f"{key}_{proj_type}"] = proj.bias[start_idx:end_idx].clone()
                            
                            # Apply bias mask
                            if self.mask_type == "scale_mask":
                                proj.bias.data[start_idx:end_idx] *= self.scale_factor
                            elif self.mask_type == "zero_mask":
                                proj.bias.data[start_idx:end_idx] = 0
        
    
    def _post_attention_hook(self, module, inputs, output, layer_idx):
        """Restore attention weights and biases after forward propagation.
        
        After the module's forward propagation is complete, restore the modified
        attention head weights and biases.
        
        Args:
            module: The current module being processed.
            inputs: The module's inputs.
            output: The module's outputs.
            layer_idx: The index of the current layer.
            
        Returns:
            The module's output.
        """
        if not hasattr(module, 'q_proj'):
            return output
            
        # Restore weights
        for key, weight in list(self.original_weights.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    if proj_type in ['q', 'k', 'v']:
                        proj.weight.data[start_idx:end_idx] = weight
                    elif proj_type == 'o':
                        proj.weight.data[:, start_idx:end_idx] = weight
                    del self.original_weights[key]
        
        # Restore biases
        for key, bias in list(self.original_biases.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name) and hasattr(getattr(module, proj_name), 'bias') and getattr(module, proj_name).bias is not None:
                    proj = getattr(module, proj_name)
                    proj.bias.data[start_idx:end_idx] = bias
                    del self.original_biases[key]
        
        return output
    
    def remove_hooks(self):
        """Remove all hooks and restore original weights and biases.
        
        Remove all hooks added to the model and ensure that all modified weights
        and biases are restored to their original state.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Ensure all original weights are restored
        for key, weight in list(self.original_weights.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.model.language_model.layers[layer_idx].self_attn
                start_idx = head_idx * self.head_dim
                end_idx = (head_idx + 1) * self.head_dim
                
                if proj_type == 'q' and hasattr(module, 'q_proj'):
                    module.q_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'k' and hasattr(module, 'k_proj'):
                    module.k_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'v' and hasattr(module, 'v_proj'):
                    module.v_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'o' and hasattr(module, 'o_proj'):
                    module.o_proj.weight.data[:, start_idx:end_idx] = weight
                
            except Exception as e:
                print(f"Error restoring weight: {key}, Error: {str(e)}")
                
        self.original_weights = {}
        
        # Ensure all original biases are restored
        for key, bias in list(self.original_biases.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.model.language_model.layers[layer_idx].self_attn
                start_idx = head_idx * self.head_dim
                end_idx = (head_idx + 1) * self.head_dim
                
                if proj_type == 'q' and hasattr(module, 'q_proj') and hasattr(module.q_proj, 'bias') and module.q_proj.bias is not None:
                    module.q_proj.bias.data[start_idx:end_idx] = bias
                elif proj_type == 'k' and hasattr(module, 'k_proj') and hasattr(module.k_proj, 'bias') and module.k_proj.bias is not None:
                    module.k_proj.bias.data[start_idx:end_idx] = bias
                elif proj_type == 'v' and hasattr(module, 'v_proj') and hasattr(module.v_proj, 'bias') and module.v_proj.bias is not None:
                    module.v_proj.bias.data[start_idx:end_idx] = bias
            except Exception as e:
                print(f"Error restoring bias: {key}, Error: {str(e)}")
                
        self.original_biases = {}
    
    # Part 2: Feature Extraction
    def apply_extraction_hooks(self, extract_k=1, clear_old=True):
        if clear_old:
            self.remove_hooks()

        self.clear_extracted_data()

        for layer_idx, layer in enumerate(self.model.language_model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                hook = layer.self_attn.o_proj.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx:
                        self._extraction_hook(module, input, output, layer_idx, extract_k=extract_k)
                )
                self.hooks.append(hook)


    def _extraction_hook(self, module, input, output, layer_idx, extract_k):
        """
        Hook function taking extract_k as an argument.
        """
        # input[0] shape: [batch, seq_len, hidden_size]
        hidden_states = input[0].detach()
        bsz, seq_len, _ = hidden_states.size()
        
        # shape: [batch, k, hidden_size]
        last_k_tokens = hidden_states[:, -extract_k:, :]
        
        # target shape: [batch, k, num_heads, head_dim]
        act_reshaped = last_k_tokens.view(bsz, extract_k, self.num_heads, self.head_dim).cpu()
        
        # Store
        if layer_idx not in self.extracted_features:
            self.extracted_features[layer_idx] = []
        self.extracted_features[layer_idx].append(act_reshaped)

    
    def get_extracted_data(self, clear=True):

        if not self.extracted_features:
            print("Warning: No features extracted")
            return None
        
        sorted_layers = sorted(self.extracted_features.keys())
        all_layers_data = []

        for l in sorted_layers:
            layer_batches = self.extracted_features[l]
            layer_tensor = torch.cat(layer_batches, dim=0)
            
            all_layers_data.append(layer_tensor.unsqueeze(1))
            
        final_tensor = torch.cat(all_layers_data, dim=1)
        
        if clear:
            self.clear_extracted_data()
        
        return final_tensor

    def clear_extracted_data(self):
        self.extracted_features = {}


class HFAttentionMasker:
    def __init__(self, model, head_mask=None, mask_type="scale_mask", scale_factor=1e-5, o_scale_factor=None):
        self.model = model
        self.head_mask = head_mask
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.o_scale_factor = o_scale_factor if o_scale_factor is not None else scale_factor
        self.hooks = []
        self.original_weights = {}
        self.extracted_features = {}
        
        self.num_heads = getattr(model.config.text_config, "num_attention_heads", 32)
        self.head_dim = getattr(model.config.text_config, "hidden_size", 4096) // self.num_heads
        self.num_layers = getattr(model.config.text_config, "num_hidden_layers")
    
    def apply_masking_hooks(self, clear_old=True):
        if self.head_mask is None:
            return
            
        if clear_old:
            self.remove_hooks()
        
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            if hasattr(layer, 'self_attn'):
                pre_hook = layer.self_attn.register_forward_pre_hook(
                    lambda module, inputs, layer_idx=layer_idx: 
                        self._pre_attention_hook(module, inputs, layer_idx)
                )
                self.hooks.append(pre_hook)
                
                post_hook = layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, layer_idx=layer_idx:
                        self._post_attention_hook(module, inputs, output, layer_idx)
                )
                self.hooks.append(post_hook)

    def _pre_attention_hook(self, module, inputs, layer_idx):
        if not hasattr(module, 'q_proj'):
            return
            
        for (masked_layer, masked_head), qkv_types in self.head_mask.items():
            if masked_layer == layer_idx:
                key = f"layer_{layer_idx}_{masked_head}"
    
                for proj_type in qkv_types:
                    proj_name = f"{proj_type}_proj"
                    if hasattr(module, proj_name):
                        proj = getattr(module, proj_name)
                        start_idx = masked_head * self.head_dim
                        end_idx = (masked_head + 1) * self.head_dim
                        
                        if proj_type == 'o':
                            self.original_weights[f"{key}_{proj_type}"] = proj.weight[:, start_idx:end_idx].clone()
                            proj.weight.data[:, start_idx:end_idx] *= self.o_scale_factor
                        else:
                            self.original_weights[f"{key}_{proj_type}"] = proj.weight[start_idx:end_idx].clone()
                            # Apply weight mask
                            if self.mask_type == "scale_mask":
                                proj.weight.data[start_idx:end_idx] *= self.scale_factor
                            elif self.mask_type == "zero_mask":
                                proj.weight.data[start_idx:end_idx] = 0
    
    def _post_attention_hook(self, module, inputs, output, layer_idx):
        if not hasattr(module, 'q_proj'):
            return output
            
        for key, weight in list(self.original_weights.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    if proj_type in ['q', 'k', 'v']:
                        proj.weight.data[start_idx:end_idx] = weight
                    elif proj_type == 'o':
                        proj.weight.data[:, start_idx:end_idx] = weight
                    del self.original_weights[key]
        
        return output
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        for key, weight in list(self.original_weights.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            module = self.model.language_model.layers[layer_idx].self_attn
            start_idx = head_idx * self.head_dim
            end_idx = (head_idx + 1) * self.head_dim
            
            if proj_type == 'q' and hasattr(module, 'q_proj'):
                module.q_proj.weight.data[start_idx:end_idx] = weight
            elif proj_type == 'k' and hasattr(module, 'k_proj'):
                module.k_proj.weight.data[start_idx:end_idx] = weight
            elif proj_type == 'v' and hasattr(module, 'v_proj'):
                module.v_proj.weight.data[start_idx:end_idx] = weight
            elif proj_type == 'o' and hasattr(module, 'o_proj'):
                module.o_proj.weight.data[:, start_idx:end_idx] = weight

        self.original_weights = {}
    

    # Part 2: Feature Extraction
    def apply_extraction_hooks(self, extract_k=1, clear_old=True):
        if clear_old:
            self.remove_hooks()

        self.clear_extracted_data()

        for layer_idx, layer in enumerate(self.model.language_model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                hook = layer.self_attn.o_proj.register_forward_hook(
                    lambda module, input, output, layer_idx=layer_idx:
                        self._extraction_hook(module, input, output, layer_idx, extract_k=extract_k)
                )
                self.hooks.append(hook)

    def _extraction_hook(self, module, input, output, layer_idx, extract_k):
        """
        Hook function taking extract_k as an argument.
        """
        # input[0] shape: [batch, seq_len, hidden_size]
        hidden_states = input[0].detach()
        bsz, seq_len, _ = hidden_states.size()
        
        # shape: [batch, k, hidden_size]
        last_k_tokens = hidden_states[:, -extract_k:, :]
        
        # target shape: [batch, k, num_heads, head_dim]
        act_reshaped = last_k_tokens.view(bsz, extract_k, self.num_heads, self.head_dim).cpu()
        
        # Store
        if layer_idx not in self.extracted_features:
            self.extracted_features[layer_idx] = []
        self.extracted_features[layer_idx].append(act_reshaped)

    
    def get_extracted_data(self, clear=True):

        if not self.extracted_features:
            print("Warning: No features extracted")
            return None
        
        sorted_layers = sorted(self.extracted_features.keys())
        all_layers_data = []

        for l in sorted_layers:
            layer_batches = self.extracted_features[l]
            layer_tensor = torch.cat(layer_batches, dim=0)
            
            all_layers_data.append(layer_tensor.unsqueeze(1))
            
        final_tensor = torch.cat(all_layers_data, dim=1)
        
        if clear:
            self.clear_extracted_data()
        
        return final_tensor

    def clear_extracted_data(self):
        self.extracted_features = {}
