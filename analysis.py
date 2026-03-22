import torch
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch.nn.functional as F

class ConflictAnalyzer:
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()
        
        if hasattr(self.model.config, "text_config"):
            self.config = self.model.config.text_config
        else:
            self.config = self.model.config
            
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        
        self._temp_buffer = {}
        self.hooks = []

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

    def clear_buffer(self):
        self._temp_buffer = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


    def get_refusal_anchors(self, benign_data, malicious_data, anchor_path="safety_anchors.pt"):
        """第一步：确立裁判 (分别计算 Attention 和 FFN 的组件级安全锚点)"""
        if os.path.exists(anchor_path):
            print(f"Loading existing component anchors from {anchor_path}...")
            return torch.load(anchor_path, map_location=self.device)

        print("Computing Component-Specific Safety Anchors (Refusal Directions) with Hooks...")

        def get_mean_module_states(dataset, desc):
            self.clear_buffer()
            self.remove_hooks()
            
            # 初始化累加器: {层号: {attn: tensor, ffn: tensor}}
            layer_sums = {l: {"attn": 0.0, "ffn": 0.0} for l in range(self.num_layers)}
            valid_count = 0

            # 注册 Hook 捕获 o_proj 和 down_proj
            for l in range(self.num_layers):
                attn_out_proj = self.model.language_model.layers[l].self_attn.o_proj
                ffn_out_proj = self.model.language_model.layers[l].mlp.down_proj

                def get_sum_hook(layer_idx, module_name):
                    def hook(module, input, output):
                        # 截取最后一个 Token
                        out_tensor = output[0] if isinstance(output, tuple) else output
                        last_token_out = out_tensor[:, -1, :].detach().cpu().squeeze(0).float()
                        # 累加到外部字典
                        layer_sums[layer_idx][module_name] += last_token_out
                    return hook

                self.hooks.append(attn_out_proj.register_forward_hook(get_sum_hook(l, "attn")))
                self.hooks.append(ffn_out_proj.register_forward_hook(get_sum_hook(l, "ffn")))

            # 前向传播收集状态
            with torch.no_grad():
                for sample in tqdm(dataset, desc=desc):
                    inputs = self.prepare_single_input(sample)
                    if inputs is None: continue
                    self.model(**inputs)
                    valid_count += 1

            self.remove_hooks()
            
            # 计算平均值
            means = {l: {"attn": layer_sums[l]["attn"] / valid_count, 
                         "ffn": layer_sums[l]["ffn"] / valid_count} for l in range(self.num_layers)}
            return means

        # 分别计算 Malicious 和 Benign 在各组件的平均隐藏状态
        mal_means = get_mean_module_states(malicious_data, "Anchor: Malicious")
        ben_means = get_mean_module_states(benign_data, "Anchor: Benign")

        # 计算差异并归一化，得到两组锚点向量
        anchors = {"attn": {}, "ffn": {}}
        for l in range(self.num_layers):
            # Attention Anchor
            diff_attn = mal_means[l]["attn"] - ben_means[l]["attn"]
            anchors["attn"][l] = (diff_attn / (torch.norm(diff_attn, p=2) + 1e-8)).to(self.device)
            
            # FFN Anchor
            diff_ffn = mal_means[l]["ffn"] - ben_means[l]["ffn"]
            anchors["ffn"][l] = (diff_ffn / (torch.norm(diff_ffn, p=2) + 1e-8)).to(self.device)

        torch.save(anchors, anchor_path)
        print(f"Component-Specific Anchors saved to {anchor_path}.")
        return anchors



    def extract_module_pushes(self, dataset, anchors, desc="Extracting Pushes"):
        """第二步 & 第三步：提取 Attention/FFN 输出，并在各自的组件级锚点上计算余弦相似度"""
        self.clear_buffer()
        self.remove_hooks()

        # 注册 Hook: 捕获 o_proj 和 down_proj 的输出
        for l in range(self.num_layers):
            attn_out_proj = self.model.language_model.layers[l].self_attn.o_proj
            ffn_out_proj = self.model.language_model.layers[l].mlp.down_proj

            def get_push_hook(layer_idx, module_name):
                def hook(module, input, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    last_token_out = out_tensor[:, -1, :].detach().float()
                    
                    # 取出对应的组件级锚点
                    anchor = anchors[module_name][layer_idx]
                    
                    # 使用余弦相似度 (Cosine Similarity)
                    push = F.cosine_similarity(last_token_out, anchor.unsqueeze(0), dim=-1).cpu()
                    
                    key = f"l{layer_idx}_{module_name}"
                    if key not in self._temp_buffer:
                        self._temp_buffer[key] = []
                    self._temp_buffer[key].append(push)
                return hook

            self.hooks.append(attn_out_proj.register_forward_hook(get_push_hook(l, "attn")))
            self.hooks.append(ffn_out_proj.register_forward_hook(get_push_hook(l, "ffn")))

        with torch.no_grad():
            for sample in tqdm(dataset, desc=desc):
                inputs = self.prepare_single_input(sample)
                if inputs is None: continue
                self.model(**inputs)

        self.remove_hooks()

        # 汇总各层的平均推力
        avg_pushes = {"attn": torch.zeros(self.num_layers), "ffn": torch.zeros(self.num_layers)}
        for l in range(self.num_layers):
            attn_pushes = torch.cat(self._temp_buffer[f"l{l}_attn"], dim=0)
            ffn_pushes = torch.cat(self._temp_buffer[f"l{l}_ffn"], dim=0)
            avg_pushes["attn"][l] = attn_pushes.mean().item()
            avg_pushes["ffn"][l] = ffn_pushes.mean().item()

        return avg_pushes


    def plot_trajectories(self, benign_pushes, malicious_pushes, jailbreak_pushes, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        layers = list(range(1, self.num_layers + 1))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        fig.suptitle("Layer-wise Directional Consistency of Components with the Refusal Anchor", fontsize=16)

        datasets = [
            ("Benign", benign_pushes, axes[0]),
            ("Malicious", malicious_pushes, axes[1]),
            ("Jailbreak", jailbreak_pushes, axes[2])
        ]

        # for title, pushes, ax in datasets:
        #     ax.plot(layers, pushes["attn"].numpy(), label="Self-Attention", color="blue", marker="o", markersize=5)
        #     ax.plot(layers, pushes["ffn"].numpy(), label="Feed-Forward (FFN)", color="red", marker="s", markersize=5)
        #     ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        #     ax.set_title(title)
        #     ax.set_xlabel("Layer Number")
        #     ax.set_ylabel("Push Magnitude (Projection)")
        #     ax.grid(True, alpha=0.3)
        #     ax.legend()

        colors = {"attn": "#1f77b4", "ffn": "#d62728"} # 经典的深蓝与深红

        for title, pushes, ax in datasets:
            ax.plot(layers, pushes["attn"].numpy(), 
                    label="Self-Attention", color=colors["attn"], 
                    marker="o", markersize=5, linewidth=1.5, alpha=0.8)
            
            ax.plot(layers, pushes["ffn"].numpy(), 
                    label="Feed-Forward (FFN)", color=colors["ffn"], 
                    marker="s", markersize=5, linewidth=1.5, alpha=0.8) # 换成方块以示区别

            # 辅助线：0刻度线代表“中立”
            ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            
            # 细节美化
            ax.set_title(title, fontweight='bold', pad=15)
            ax.set_xlabel("Transformer Layer Index")
            if ax == axes[0]:
                ax.set_ylabel("Cosine Similarity")
            
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xticks(np.arange(0, self.num_layers + 1, 5)) # 每隔5层显示一个刻度
            
            # 只有最后一张图显示图例，避免视觉干扰
            if ax == axes[2]:
                ax.legend(frameon=True, loc='upper right', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "module_conflict_trajectory.png")
        plt.savefig(save_path, dpi=300)
        print(f"\nAwesome! The trajectory plot is saved to {save_path}")

    def run_experiment(self, benign_data, malicious_data, jailbreak_data, output_dir):
        print("\n=== Phase 1: Establishing the Refusal Anchor ===")
        anchors = self.get_refusal_anchors(benign_data, malicious_data, anchor_path=os.path.join(output_dir, "anchors.pt"))

        print("\n=== Phase 2 & 3: Extracting Evidence and Computing Push ===")
        benign_pushes = self.extract_module_pushes(benign_data, anchors, desc="Processing Benign")
        mal_pushes = self.extract_module_pushes(malicious_data, anchors, desc="Processing Malicious")
        jail_pushes = self.extract_module_pushes(jailbreak_data, anchors, desc="Processing Jailbreak")

        print("\n=== Phase 4: Visualizing the Aha! Moment ===")
        self.plot_trajectories(benign_pushes, mal_pushes, jail_pushes, output_dir)


if __name__ == "__main__":
    # 配置你的路径
    MODEL_PATH = "/home/zengxiyu24/mllms_know/llava-v1.6-vicuna-7b-hf"
    DATA_DIR = "/home/zengxiyu24/mllms_know/collected_samples/llava"
    OUTPUT_DIR = "validation/llava/results"
    
    BENIGN_PATH = os.path.join(DATA_DIR, "benign_bunny.json")
    MALICIOUS_PATH = os.path.join(DATA_DIR, "malicious_guard.json") 
    JAILBREAK_PATH = os.path.join(DATA_DIR, "jailbreak_guard.json") 

    def load_json(path, name):
        if not os.path.exists(path):
            print(f"Error: {name} dataset not found at {path}")
            return []
        with open(path, 'r') as f:
            data = json.load(f)

        return data

    print(f"Loading model from {MODEL_PATH} ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
    
    analyzer = ConflictAnalyzer(model, processor)

    benign_data = load_json(BENIGN_PATH, "Benign")
    malicious_data = load_json(MALICIOUS_PATH, "Malicious")
    jailbreak_data = load_json(JAILBREAK_PATH, "Jailbreak")

    if benign_data and malicious_data and jailbreak_data:
        analyzer.run_experiment(benign_data, malicious_data, jailbreak_data, OUTPUT_DIR)
    else:
        print("Datasets missing. Please check file paths.")