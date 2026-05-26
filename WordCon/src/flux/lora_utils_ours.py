import os
import torch
from safetensors.torch import load_file
import logging
import copy

def load_lora(pipe, lora_path, adapter_name=None, alpha=None, use_adapter_method=True):
    """
    Load LoRA weights into the model
    
    Args:
        pipe: Model pipeline or transformer
        lora_path: Path to the LoRA weights file
        adapter_name: Adapter name, required when loading with adapter method
        alpha: LoRA scaling factor, used in manual loading. If None, calculated automatically based on rank
        use_adapter_method: Whether to use adapter method to load. If False, use manual loading
        
    Returns:
        Model with loaded LoRA weights
    """
    logging.info(f"Loading LoRA weights: {lora_path}")
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")
    
    # Check current device of the model
    device = pipe.device
    
    # Method 1: Load using adapter method
    if use_adapter_method:
        if not adapter_name:
            raise ValueError("adapter_name parameter must be provided when using adapter method")
        
        try:
            # Try to use load_lora_weights method
            if hasattr(pipe, 'load_lora_weights'):
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                logging.info(f"Loaded LoRA weights using load_lora_weights method, adapter name: {adapter_name}")
            # Try to use load_lora method
            elif hasattr(pipe, 'load_lora'):
                pipe.load_lora(lora_path)
                logging.info("Loaded LoRA weights using load_lora method")
            else:
                logging.warning("Model does not have built-in LoRA loading method, switching to manual loading")
                use_adapter_method = False
        except Exception as e:
            logging.warning(f"Failed to load using adapter method: {str(e)}, switching to manual loading")
            use_adapter_method = False
    
    # Method 2: Manually load and apply LoRA weights
    if not use_adapter_method:
        # Load safetensors file
        lora_state_dict = load_file(lora_path)
        
        # Get original model state dict
        transformer = pipe.transformer if hasattr(pipe, 'transformer') else pipe
        model_state_dict = transformer.state_dict()
        
        # Parse LoRA weights and apply
        lora_layer_mapping = {}
        
        # Step 1: Organize LoRA weight pairs
        for key in lora_state_dict.keys():
            if 'lora_A' in key or 'lora_B' in key:
                # Extract module path, remove the first "transformer." prefix
                if key.startswith("transformer."):
                    # Remove the first "transformer." prefix
                    module_path = key[len("transformer."):key.rindex('.lora_')]
                    layer_type = 'A' if 'lora_A' in key else 'B'
                    
                    if module_path not in lora_layer_mapping:
                        lora_layer_mapping[module_path] = {}
                    
                    # Move weights to the same device as the model
                    lora_layer_mapping[module_path][layer_type] = lora_state_dict[key].to(device)
                else:
                    logging.warning(f"Warning: Key name does not match expected format: {key}")
        
        # Detect LoRA rank
        detected_rank = None
        for module_path, matrices in lora_layer_mapping.items():
            if 'A' in matrices:
                detected_rank = matrices['A'].shape[0]
                break
        
        if detected_rank is None:
            raise ValueError("Unable to detect LoRA rank. Please check if the LoRA weights file is correct.")
        else:
            # If alpha is not provided, calculate automatically based on rank
            if alpha is None:
                alpha = detected_rank
                logging.info(f"Detected LoRA rank as {detected_rank}, automatically set alpha={alpha}")
        
        logging.info(f"Found {len(lora_layer_mapping)} LoRA layers")
        
        # Step 2: Apply LoRA updates
        for module_path, matrices in lora_layer_mapping.items():
            if 'A' in matrices and 'B' in matrices:
                # Build target key name
                target_key = module_path + ".weight"  # Assuming all targets are weight matrices
                
                # Check if target key exists in model state dict
                if target_key in model_state_dict:
                    # Get original weight
                    original_weight = model_state_dict[target_key]
                    
                    # Calculate LoRA update: W + alpha * (B * A)
                    lora_A = matrices['A']
                    lora_B = matrices['B']
                    
                    # Ensure dimensions match
                    if lora_B.shape[0] == original_weight.shape[0] and lora_A.shape[1] == original_weight.shape[1]:
                        # Calculate low-rank update
                        lora_update = torch.matmul(lora_B, lora_A)
                        
                        # Apply scaling factor
                        rank = lora_A.shape[0]  # LoRA rank
                        lora_update = lora_update * (alpha / rank)
                        
                        # Update weight
                        model_state_dict[target_key] = original_weight + lora_update
                    else:
                        logging.warning(f"Dimension mismatch: {target_key}, Original: {original_weight.shape}, "
                                        f"LoRA B: {lora_B.shape}, LoRA A: {lora_A.shape}")
                else:
                    logging.warning(f"Target parameter not found: {target_key}")
        
        # Load the merged state dict
        transformer.load_state_dict(model_state_dict)
        logging.info("Manual LoRA weights loading completed")
    
    return pipe