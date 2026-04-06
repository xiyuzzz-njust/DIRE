import torch
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch.nn.functional as F

import sys
sys.path.append('')
from utils.neuron_selector import NeuronManager

class CausalNeuronSelector:
    def __init__(self, model, processor, neuron_manager, device='cuda'):
        self.model = model
        self.processor = processor
        self.neuron_manager = neuron_manager
        self.device = device
        self.model.eval()
        
        if hasattr(self.model.config, "text_config"):
            self.config = self.model.config.text_config
        else:
            self.config = self.model.config
            
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size

    def prepare_single_input(self, sample):
        text_prompt = sample.get('prompt')
        image_path = sample.get('image_path')
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            return inputs
        return None


    def get_effective_layers_and_anchors(self, benign_data, malicious_data, save_path, drop_ratio=0.3):
        if os.path.exists(save_path):
            print(f"Loading effective layers and FFN-specific anchors from {save_path}...")
            return torch.load(save_path, map_location=self.device)

        print("Computing Macro-Separation (Residual) and Micro-Anchors (FFN) in one pass...")

        def get_layer_stats(dataset, desc):
            resid_sums = {l: 0.0 for l in range(self.num_layers)}
            ffn_sums = {l: 0.0 for l in range(self.num_layers)}
            valid_count = 0
            
            hooks = []
            for l in range(self.num_layers):
                module = self.model.language_model.layers[l].mlp.down_proj
                def get_hook(layer_idx):
                    def hook(mod, inp, out):
                        out_tensor = out[0] if isinstance(out, tuple) else out
  
                        ffn_sums[layer_idx] += out_tensor[:, -1, :].detach().cpu().squeeze(0).float()
                    return hook
                hooks.append(module.register_forward_hook(get_hook(l)))

            with torch.no_grad():
                for sample in tqdm(dataset, desc=desc):
                    inputs = self.prepare_single_input(sample)
                    if inputs is None: continue
                    
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    for l in range(self.num_layers):
                        last_token_h = outputs.hidden_states[l+1][:, -1, :].detach().cpu().squeeze(0).float()
                        resid_sums[l] += last_token_h
                        
                    valid_count += 1
            
            for h in hooks: 
                h.remove()
                
            mean_resid = {l: resid_sums[l] / valid_count for l in range(self.num_layers)}
            mean_ffn = {l: ffn_sums[l] / valid_count for l in range(self.num_layers)}
            
            return mean_resid, mean_ffn

        mal_resid, mal_ffn = get_layer_stats(malicious_data, "Forward Pass: Malicious")
        ben_resid, ben_ffn = get_layer_stats(benign_data, "Forward Pass: Benign")

        separations = {}
        anchors = {}
        for l in range(self.num_layers):

            resid_diff = mal_resid[l] - ben_resid[l]
            sep = torch.norm(resid_diff, p=2).item()
            separations[l] = sep

            ffn_diff = mal_ffn[l] - ben_ffn[l]
            anchors[l] = ffn_diff / (torch.norm(ffn_diff, p=2) + 1e-8)

        sorted_layers = sorted(separations.items(), key=lambda x: x[1], reverse=True)
        keep_count = int(self.num_layers * (1 - drop_ratio))
        L_eff = [layer_idx for layer_idx, sep in sorted_layers[:keep_count]]
        L_eff.sort()
        
        print(f"\nDropping bottom {drop_ratio*100}% layers.")
        print(f"Identified {len(L_eff)} Effective Layers (L_eff): {L_eff}")

        data_to_save = {"L_eff": L_eff, "anchors": anchors, "separations": separations}
        torch.save(data_to_save, save_path)
        return data_to_save
    

    def capture_activations(self, data_configs):

        print("\n--- Phase 2a: Capturing FFN Activations ---")
        self.neuron_manager.register_hooks(hook_target="down")
        
        try:
            with torch.no_grad():
                for cfg in data_configs:
                    dataset = cfg["data"]
                    label = cfg["label"]
                    for i, sample in enumerate(tqdm(dataset, desc=f"Scanning {cfg['desc']}")):
                        inputs = self.prepare_single_input(sample)
                        if inputs is not None:
                            self.model(**inputs)
                        if i % 20 == 0: torch.cuda.empty_cache()
                    
                    self.neuron_manager.flush_buffer_to_storage(label)
        finally:
            self.neuron_manager.remove_hooks()
            print("Neuron capture complete. Hooks removed.")

    def analyze_causal_neurons(self, L_eff, anchors, top_k=1000, save_path="causal_neurons.json"):
        print("\n--- Phase 2b: Calculating Causal Pushes (P_ffn) ---")
        
        results_robust = []
        results_vuln = []

        def rms_norm_pt(tensor, eps=1e-6):
            variance = tensor.pow(2).mean(-1, keepdim=True)
            return tensor * torch.rsqrt(variance + eps)

        def rms_norm_np(arr, eps=1e-6):
            variance = np.mean(arr**2, axis=0, keepdims=True)
            return arr / np.sqrt(variance + eps)

        for l in tqdm(L_eff, desc="Analyzing Effective Layers"):
            v_dir = anchors[l].to(self.device).squeeze()
            
            w_down = self.model.language_model.layers[l].mlp.down_proj.weight.data
            geo_push = torch.matmul(w_down.T.float(), v_dir.float()).cpu().numpy()


            act_mal = np.vstack(self.neuron_manager.captured_data['malicious'][l])
            act_ben = np.vstack(self.neuron_manager.captured_data['benign'][l])
            act_jail = np.vstack(self.neuron_manager.captured_data['jailbreak'][l])

            E_act_mal = np.mean(act_mal, axis=0)
            E_act_ben = np.mean(act_ben, axis=0)
            E_act_jail = np.mean(act_jail, axis=0)


            E_P_mal = E_act_mal * geo_push
            E_P_ben = E_act_ben * geo_push
            E_P_jail = E_act_jail * geo_push

            for i in range(self.intermediate_size):
                if E_P_mal[i].item() > 0 and E_P_jail[i].item() > 0:
                    s_robust = float(E_P_mal[i] + E_P_jail[i] - 2 * E_P_ben[i])
                    results_robust.append({"layer": l, "neuron": i, "score": s_robust, "E_mal_push": float(E_P_mal[i])})

                if E_P_mal[i].item() > 0:
                    s_vuln = float(E_P_mal[i] - E_P_jail[i])
                    results_vuln.append({"layer": l, "neuron": i, "score": s_vuln, "E_mal_push": float(E_P_mal[i]), "E_jail_push": float(E_P_jail[i])})


        results_robust.sort(key=lambda x: x['score'], reverse=True)
        results_vuln.sort(key=lambda x: x['score'], reverse=True)
        
        top_robust = results_robust[:top_k]
        top_vuln = results_vuln[:top_k]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({
                "meta": {
                    "L_eff": L_eff,
                    "top_k_selected": top_k,
                },
                "robust_neurons": {"neurons": top_robust},
                "vulnerable_neurons": {"neurons": top_vuln}
            }, f, indent=4)
            
        print(f"\nSuccess! Found Top {top_k} Causal FFN Neurons.")
        print(f"Results saved to {save_path}")

def load_json(path, name):
    if not os.path.exists(path):
        print(f"Error: {name} dataset not found at {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    MODEL_PATH = "llava-v1.6-vicuna-7b-hf"
    DATA_DIR = ""
    OUTPUT_DIR = "validation/llava/results"
    
    BENIGN_PATH = os.path.join(DATA_DIR, "benign_bunny.json")
    MALICIOUS_PATH = os.path.join(DATA_DIR, "malicious_guard.json") 
    JAILBREAK_PATH = os.path.join(DATA_DIR, "jailbreak_guard.json") 

    print(f"Loading model from {MODEL_PATH} ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
    

    neuron_manager = NeuronManager(model)
    selector = CausalNeuronSelector(model, processor, neuron_manager)

    benign_data = load_json(BENIGN_PATH, "Benign")
    malicious_data = load_json(MALICIOUS_PATH, "Malicious")
    jailbreak_data = load_json(JAILBREAK_PATH, "Jailbreak")


    drop_ratio = 0.1
    anchor_path = os.path.join(OUTPUT_DIR, f"effective_anchors_neurons_{drop_ratio}.pt")
    stage1_data = selector.get_effective_layers_and_anchors(
        benign_data, malicious_data, save_path=anchor_path, drop_ratio=drop_ratio
    )


    data_configs = [
        {"data": jailbreak_data, "label": "jailbreak", "desc": "Jailbreak Data"},
        {"data": malicious_data, "label": "malicious", "desc": "Malicious(Refusal) Data"}, 
        {"data": benign_data,    "label": "benign",    "desc": "Benign Data"}             
    ]
    selector.capture_activations(data_configs)


    save_path = os.path.join(OUTPUT_DIR, f"causal_neurons_all(three)_{drop_ratio}.json")
    selector.analyze_causal_neurons(
        L_eff=stage1_data["L_eff"], 
        anchors=stage1_data["anchors"], 
        top_k=5000, 
        save_path=save_path
    )