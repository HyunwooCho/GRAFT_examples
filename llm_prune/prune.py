import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import prune

class PrunedLLM:
    def __init__(self, model_name, device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def apply_unstructured_pruning(self, amount=0.3):
        """ Apply Unstructured Pruning to Linear Layers """
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')  # Remove pruned weights
    
    def apply_structured_pruning(self, amount=0.3):
        """ Apply Structured Pruning to Linear Layers (Neuron-level) """
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=1)
                prune.remove(module, 'weight')  # Remove pruned weights
    
    def quantize_model(self):
        """ Apply Dynamic Quantization """
        self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
    
    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description="Prune and Quantize LLM")
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b", help="Pretrained model name")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--unstructured", type=float, default=0.3, help="Unstructured pruning amount")
    parser.add_argument("--structured", type=float, default=0.3, help="Structured pruning amount")
    parser.add_argument("--save_path", type=str, default="pruned_sllm", help="Path to save pruned model")
    parser.add_argument("--test_prompt", type=str, default="Once upon a time,", help="Prompt for testing the model")
    args = parser.parse_args()
    
    llm = PrunedLLM(args.model_name, args.device)
    llm.apply_unstructured_pruning(args.unstructured)
    llm.apply_structured_pruning(args.structured)
    llm.quantize_model()
    
    print("Generated Text:", llm.generate_text(args.test_prompt))
    
    llm.save_model(args.save_path)
    print(f"Model saved to {args.save_path}")
    

if __name__ == "__main__":
    main()
