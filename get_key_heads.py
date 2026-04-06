import torch
import json
import os
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import torch.nn.functional as F

import sys
sys.path.append('')
from utils.masker import AttentionMasker

class CausalHeadSelector:
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        self.analyzer = AttentionMasker(model)
        self.model.eval()
        
        if hasattr(self.model.config, "text_config"):
            self.config = self.model.config.text_config
        else:
            self.config = self.model.config
            
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.hidden_size = self.config.hidden_size
        
        print(f"Initialized Causal Selector. Layers: {self.num_layers}, Heads: {self.num_heads}")

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
            print(f"Loading effective layers and ATTN-specific anchors from {save_path}...")
            return torch.load(save_path, map_location=self.device)

        print("Computing Macro-Separation (Residual Stream) and Micro-Anchors (Attention) in one pass...")

        def get_layer_stats(dataset, desc):

            resid_sums = {l: 0.0 for l in range(self.num_layers)}
   
            attn_sums = {l: 0.0 for l in range(self.num_layers)}
            valid_count = 0
            
            hooks = []

            for l in range(self.num_layers):
                module = self.model.language_model.layers[l].self_attn.o_proj
                def get_hook(layer_idx):
                    def hook(mod, inp, out):
                        out_tensor = out[0] if isinstance(out, tuple) else out

                        attn_sums[layer_idx] += out_tensor[:, -1, :].detach().cpu().squeeze(0).float()
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
            mean_attn = {l: attn_sums[l] / valid_count for l in range(self.num_layers)}
            
            return mean_resid, mean_attn

        mal_resid, mal_attn = get_layer_stats(malicious_data, "Forward Pass: Malicious")
        ben_resid, ben_attn = get_layer_stats(benign_data, "Forward Pass: Benign")

        separations = {}
        anchors = {}
        
        for l in range(self.num_layers):
            resid_diff = mal_resid[l] - ben_resid[l]
            sep = torch.norm(resid_diff, p=2).item()
            separations[l] = sep

            attn_diff = mal_attn[l] - ben_attn[l]
            anchors[l] = attn_diff / (torch.norm(attn_diff, p=2) + 1e-8)

        sorted_layers = sorted(separations.items(), key=lambda x: x[1], reverse=True)
        keep_count = int(self.num_layers * (1 - drop_ratio))

        L_eff = [layer_idx for layer_idx, sep in sorted_layers[:keep_count]]
        L_eff.sort()
        
        dropped_layers = [layer_idx for layer_idx, sep in sorted_layers[keep_count:]]
        dropped_layers.sort()
        print(f"Dropped {len(dropped_layers)} Layers: {dropped_layers}")
        print(f"Identified {len(L_eff)} Effective Layers (L_eff): {L_eff}")

        data_to_save = {
            "L_eff": L_eff,
            "anchors": anchors,
            "separations": separations
        }
        torch.save(data_to_save, save_path)
        print(f"Data saved to {save_path}.")
        return data_to_save
    

    def compute_head_pushes(self, dataset, L_eff, anchors, desc):
        self.analyzer.apply_extraction_hooks(extract_k=1)
        valid_count = 0
        with torch.no_grad():
            for sample in tqdm(dataset, desc=desc):
                inputs = self.prepare_single_input(sample)
                if inputs is None: continue
                self.model(**inputs)
                valid_count += 1
                if valid_count % 10 == 0:
                    torch.cuda.empty_cache()

        raw_data = self.analyzer.get_extracted_data(clear=True) # [B, L, 1, H, D]
        self.analyzer.remove_hooks()
        raw_data = raw_data.squeeze(2).to(self.device) # [B, L, H, D]

        B = raw_data.shape[0]
        pushes = torch.zeros(B, self.num_layers, self.num_heads).to(self.device)

        print(f"Calculating Projection Pushes for {desc} in L_eff...")
        for l in L_eff:
            anchor_v = anchors[l].squeeze(0).to(self.device) # [hidden_size]
            o_proj_weight = self.model.language_model.layers[l].self_attn.o_proj.weight # [hidden_size, hidden_size]
            
            for h in range(self.num_heads):
                start_idx = h * self.head_dim
                end_idx = (h + 1) * self.head_dim
                
                W_O_h = o_proj_weight[:, start_idx:end_idx]

                x_h = raw_data[:, l, h, :]
                out_h = torch.matmul(x_h, W_O_h.T)

                out_h_norm = self.rms_norm(out_h)
                push_h = torch.matmul(out_h_norm.float(), anchor_v) # [B]

                
                pushes[:, l, h] = push_h

        return pushes.cpu() 

    def rms_norm(self, tensor, eps=1e-6):
        variance = tensor.pow(2).mean(-1, keepdim=True)
        return tensor * torch.rsqrt(variance + eps)

    def run_causal_screening(self, benign_data, malicious_data, jailbreak_data, output_dir, top_k=50, drop_ratio=0.3):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n--- Phase 1: Semantic Separation & Effective Layers ---")
        anchor_path = os.path.join(output_dir, f"effective_anchors_attn_{drop_ratio}.pt")
        stage1_data = self.get_effective_layers_and_anchors(benign_data, malicious_data, anchor_path, drop_ratio=drop_ratio)
        L_eff = stage1_data["L_eff"]
        anchors = stage1_data["anchors"]

        print("\n--- Phase 2: Extracting Pushes for Causal Screening ---")
        P_benign = self.compute_head_pushes(benign_data, L_eff, anchors, "Benign")
        P_mal = self.compute_head_pushes(malicious_data, L_eff, anchors, "Malicious")
        P_jail = self.compute_head_pushes(jailbreak_data, L_eff, anchors, "Jailbreak")

        E_benign = P_benign.mean(dim=0)
        E_mal = P_mal.mean(dim=0)
        E_jail = P_jail.mean(dim=0)

        print("\n--- Phase 3: Scoring & Sorting ---")
        results_robust = []
        results_vuln = []

        for l in L_eff:
            for h in range(self.num_heads):

                if E_mal[l, h].item() > 0 and E_jail[l, h].item() > 0:
                    s_robust = (E_mal[l, h] + E_jail[l, h] - 2 * E_benign[l, h]).item()
                    results_robust.append({"layer": l, "head": h, "score": s_robust, "E_mal": E_mal[l,h].item()})


                if E_mal[l, h].item() > 0:
                    s_vuln = (E_mal[l, h] - E_jail[l, h]).item()
                    results_vuln.append({"layer": l, "head": h, "score": s_vuln})

        results_robust.sort(key=lambda x: x['score'], reverse=True)
        results_vuln.sort(key=lambda x: x['score'], reverse=True)

        top_robust = results_robust[:top_k]
        top_vuln = results_vuln[:top_k]

        output_json = os.path.join(output_dir, f"causal_attention_heads_all(three)_{drop_ratio}.json")
        with open(output_json, 'w') as f:
            json.dump({
                "L_eff": L_eff,
                "top_k_selected": top_k,
                "robust_heads": top_robust,
                "vulnerable_heads": top_vuln
            }, f, indent=4)
            
        print(f"\nSuccess! Filtered Attention Heads in Effective Layers ({len(L_eff)} layers).")
        print(f"Results saved to {output_json}")


if __name__ == "__main__":
    MODEL_PATH = "llava-v1.6-vicuna-7b-hf"
    DATA_DIR = ""
    OUTPUT_DIR = "validation/llava/results"
    
    BENIGN_PATH = os.path.join(DATA_DIR, "benign_bunny.json")
    MALICIOUS_PATH = os.path.join(DATA_DIR, "malicious_guard.json") 
    JAILBREAK_PATH = os.path.join(DATA_DIR, "jailbreak_guard.json") 

    def load_json(path, name):
        if not os.path.exists(path):
            print(f"Error: {name} dataset not found at {path}")
            return []
        with open(path, 'r') as f:
            return json.load(f)

    print(f"Loading model from {MODEL_PATH} ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
    
    selector = CausalHeadSelector(model, processor)

    benign_data = load_json(BENIGN_PATH, "Benign")
    malicious_data = load_json(MALICIOUS_PATH, "Malicious")
    jailbreak_data = load_json(JAILBREAK_PATH, "Jailbreak")

    if benign_data and malicious_data and jailbreak_data:
        selector.run_causal_screening(benign_data, malicious_data, jailbreak_data, OUTPUT_DIR, top_k=50, drop_ratio=0.1)