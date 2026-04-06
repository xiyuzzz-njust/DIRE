import torch
import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import math

import sys
sys.path.append("") 
from models.llava import LLaVA
from utils.visualizer import analyze_and_save_results

import time

def load_json_list(path, key_path, top_k):
    if not path or not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    for key in key_path:
        data = data.get(key, data)
    return data[:top_k]

class ThreatDetector:
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper.model
        self.processor = model_wrapper.processor
        self.device = self.model.device
        
        config = self.model.config.get_text_config() if hasattr(self.model.config, "text_config") else self.model.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // config.num_attention_heads
        
        self.use_attn = args.attn_json_path is not None
        self.use_ffn = args.ffn_json_path is not None
        
        if not self.use_attn and not self.use_ffn:
            raise ValueError("Error: You must provide at least one of --attn_json_path or --ffn_json_path!")

        print(">>> Loading Component-Specific Anchors...")
        attn_stage1 = torch.load(args.attn_anchor_path, map_location=self.device)
        self.L_eff = attn_stage1["L_eff"]
        self.attn_anchors = attn_stage1["anchors"]
        
        ffn_stage1 = torch.load(args.ffn_anchor_path, map_location=self.device)
        self.ffn_anchors = ffn_stage1["anchors"]

        separations = attn_stage1["separations"] 
        eff_seps = [separations[l] for l in self.L_eff]
        eff_tensor = torch.tensor(eff_seps, dtype=torch.float32)
        eff_norm = (eff_tensor - eff_tensor.mean()) / (eff_tensor.std() + 1e-6)
        T = args.temperature 
        weights = torch.nn.functional.softmax(eff_norm / T, dim=0)
        self.layer_weights = {l: weights[i].item() for i, l in enumerate(self.L_eff)}
        

        self.rob_attn = load_json_list(args.attn_json_path, ['robust_heads'], args.attn_top_k) if self.use_attn else []
        self.rob_ffn = load_json_list(args.ffn_json_path, ['robust_neurons', 'neurons'], args.ffn_top_k) if self.use_ffn else []

        
        self._prepare_component_masks()
        self.hooks = []
        self.layer_vectors = {}

    def _prepare_component_masks(self):
        print(f">>> Preparing Component Masks (Attn: {self.use_attn}, FFN: {self.use_ffn})...")
        self.masks = {
            "R_attn": {l: torch.zeros(self.hidden_size, device=self.device) for l in self.L_eff},
            "R_ffn": {l: torch.zeros(self.model.config.text_config.intermediate_size, device=self.device) for l in self.L_eff},

        }

        for item in self.rob_attn:
            l, h = item['layer'], item['head']
            if l in self.L_eff: self.masks["R_attn"][l][h * self.head_dim : (h + 1) * self.head_dim] = 1.0

        for item in self.rob_ffn:
            l, n = item['layer'], item['neuron']
            if l in self.L_eff: self.masks["R_ffn"][l][n] = 1.0
            

    def register_threat_hooks(self):
        self.remove_hooks()
        
        def get_attn_hook(l):
            w_o_t = self.model.language_model.layers[l].self_attn.o_proj.weight.data.T.float()
            def hook(module, input, output):
                x = input[0][:, -1, :].detach().float().squeeze(0)
                self.layer_vectors[l]["R_attn"] = torch.matmul(x * self.masks["R_attn"][l], w_o_t)
            return hook

        def get_ffn_hook(l):
            w_down_t = self.model.language_model.layers[l].mlp.down_proj.weight.data.T.float()
            def hook(module, input, output):
                a = input[0][:, -1, :].detach().float().squeeze(0)
                self.layer_vectors[l]["R_ffn"] = torch.matmul(a * self.masks["R_ffn"][l], w_down_t)
            return hook

        for l in self.L_eff:
            layer_module = self.model.language_model.layers[l]
            if self.use_attn:
                self.hooks.append(layer_module.self_attn.o_proj.register_forward_hook(get_attn_hook(l)))
            if self.use_ffn:
                self.hooks.append(layer_module.mlp.down_proj.register_forward_hook(get_ffn_hook(l)))

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def rms_norm_pt(self, tensor, eps=1e-6):
        variance = tensor.pow(2).mean(-1, keepdim=True)
        return tensor * torch.rsqrt(variance + eps)

    def get_threat_score(self, prompt_text, image_path):
        for l in self.L_eff:
            zero_vec = torch.zeros(self.hidden_size, device=self.device)
            self.layer_vectors[l] = {
                "R_attn": zero_vec, 
                "R_ffn": zero_vec
            }
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=Image.open(image_path).convert("RGB"), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            self.model(**inputs) 
            
        threat_score_final = 0.0
        raw_proj_R_attn = 0.0    
        raw_proj_R_ffn = 0.0      

        for l in self.L_eff:
            anchor_attn_l = self.attn_anchors[l].squeeze().float()
            anchor_ffn_l = self.ffn_anchors[l].squeeze().float()

            vec_R_attn = self.layer_vectors[l]["R_attn"]
            vec_R_ffn = self.layer_vectors[l]["R_ffn"]


            vec_R_attn_norm = self.rms_norm_pt(vec_R_attn)
            vec_R_ffn_norm = self.rms_norm_pt(vec_R_ffn)

            proj_R_attn = torch.dot(vec_R_attn_norm, anchor_attn_l).item()
            proj_R_ffn = torch.dot(vec_R_ffn_norm, anchor_ffn_l).item()


            weight_l = self.layer_weights[l]

            raw_proj_R_attn += proj_R_attn * weight_l
            raw_proj_R_ffn += proj_R_ffn * weight_l



            if self.use_attn and self.use_ffn:

                r_attn_pos = max(0.0, proj_R_attn)
                r_ffn_pos = max(0.0, proj_R_ffn)

                score_l = 2 * r_attn_pos * r_ffn_pos / (r_attn_pos + r_ffn_pos + 1e-6)

            elif self.use_attn:
                score_l = proj_R_attn
            elif self.use_ffn:
                score_l = proj_R_ffn
            else:
                score_l = 0.0
                
            threat_score_final += score_l * weight_l

        return {
            "threat_score": threat_score_final,
            "raw_proj_R_attn_score": raw_proj_R_attn,
            "raw_proj_R_ffn_score": raw_proj_R_ffn,
        }

def load_samples(path, limit=100):
    if not os.path.exists(path): return []
    with open(path, 'r') as f: return json.load(f)[:limit]

def run_evaluation(args):
    print(">>> Initializing Model and Threat Detector...")
    model_wrapper = LLaVA(args.model_path)
    detector = ThreatDetector(model_wrapper, args)
    
    benign_samples = load_samples(os.path.join(args.data_dir, "mm_vet.json"), 100) + \
                     load_samples(os.path.join(args.data_dir, "bunny.json"), 100)
    
    malicious_datasets = {
        "HarmBench": load_samples(os.path.join(args.data_dir, "harmbench.json"), 106),
        "Query_Relevant": load_samples(os.path.join(args.data_dir, "Query_Relevant.json"), 200),
        "Figstep": load_samples(os.path.join(args.data_dir, "Figstep.json"), 200),
        "Spa_vl": load_samples(os.path.join(args.data_dir, "Spa_vl.json"), 200)
    }

    results = []
    detector.register_threat_hooks()

    try:

        for s in tqdm(benign_samples, desc="Evaluating Benign (Label 0)"):
            scores = detector.get_threat_score(s.get('prompt'), s.get('image_path'))

            scores.update({"dataset": "Benign", "label": 0})
            results.append(scores)
        

        for name, samples in malicious_datasets.items():
            for s in tqdm(samples, desc=f"Evaluating {name} (Label 1)"):
                scores = detector.get_threat_score(s.get('prompt'), s.get('image_path'))
                scores.update({"dataset": name, "label": 1})
                results.append(scores)
    finally:
        detector.remove_hooks()

    file_prefix = f"threat_tem{args.temperature}"
    if args.attn_json_path:
        file_prefix += f"_topA{args.attn_top_k}"
    if args.ffn_json_path:
        file_prefix += f"_topF{args.ffn_top_k}"
        
    if args.suffix_name:
        file_prefix += f"_{args.suffix_name}"
    
    final_metrics = analyze_and_save_results(
        results=results, 
        output_dir=args.output_dir, 
        file_prefix=file_prefix
    )

    print("\nEvaluation Complete! Check the output directory for JSON results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/zengxiyu24/mllms_know/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="threat_detection_results")
    
    parser.add_argument("--attn_anchor_path", type=str, required=True, help="Path to effective_layers_and_attn_anchors.pt")
    parser.add_argument("--ffn_anchor_path", type=str, required=True, help="Path to effective_layers_and_ffn_anchors.pt")
    
    parser.add_argument("--attn_json_path", type=str, default=None, help="Path to causal_attn_screening.json")
    parser.add_argument("--ffn_json_path", type=str, default=None, help="Path to causal_critical_neurons.json")
    
    parser.add_argument("--attn_top_k", type=int, default=50)
    parser.add_argument("--ffn_top_k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--suffix_name", type=str, default=None, help="Optional suffix for output files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    run_evaluation(args)