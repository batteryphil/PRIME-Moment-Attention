"""
PRIME-Scout: Local Repository Evaluator
Evaluates candidate repositories using local LLM (Qwen2.5-Coder-1.5B on ROCm/GPU)
grounded in sandbox execution telemetry, AST audits, and research alignment.
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional
from scout.config import MODEL_CONFIG, RESEARCH_PROFILE

class PrimeScoutEvaluator:
    def __init__(self, mode: str = "auto", device: Optional[str] = None):
        """
        mode: 'llm' (loads local Qwen2.5-Coder), 'heuristic' (fast rules-based), or 'auto'
        """
        self.device = device or MODEL_CONFIG["device"]
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def load_model(self):
        if self._initialized:
            return
        
        print(f"[*] PRIME-Scout: Loading local model {MODEL_CONFIG['model_id']} on {self.device}...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_id"])
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["model_id"],
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.model.to(self.device)
            
        self.model.eval()
        self._initialized = True
        print(f"[+] Local model successfully initialized on {self.device}!")

    def evaluate(self, repo: Dict[str, Any], readme_text: str, sandbox_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a repository using either LLM or fast heuristic fallback.
        """
        if self.mode == "heuristic":
            return self._heuristic_evaluate(repo, readme_text, sandbox_res)

        try:
            if not self._initialized:
                self.load_model()
            return self._llm_evaluate(repo, readme_text, sandbox_res)
        except Exception as e:
            print(f"[!] LLM evaluation encountered error: {e}. Falling back to heuristic scoring.")
            return self._heuristic_evaluate(repo, readme_text, sandbox_res)

    def _llm_evaluate(self, repo: Dict[str, Any], readme_text: str, sandbox_res: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        
        audit = sandbox_res.get("audit", {})
        sandbox_status = sandbox_res.get("status", "SKIPPED")
        sandbox_log = sandbox_res.get("log", "No log.")
        
        readme_snippet = readme_text[:4000].strip() if readme_text else "No README available."
        
        prompt = f"""<|im_start|>system
You are PRIME-Scout, an expert AI research evaluator assisting batteryphil, the author of PRIME-Moment-Attention (a constant-memory 2nd-order moment attention mechanism on ROCm/AMD GPUs).
Evaluate the candidate repository for research synergy, code viability, and architectural innovation.
Target domains: Linear attention, ROCm/HIP kernels, Triton optimization, KV cache compression, Decision Transformers, and Generative Video.
Respond ONLY with a valid JSON object.
<|im_end|>
<|im_start|>user
Candidate Repository:
- Name: {repo.get('full_name', repo.get('name'))}
- URL: {repo.get('repo_url')}
- Stars: {repo.get('stars', 0):,}
- Topics: {repo.get('topics', [])}
- Description: {repo.get('description', 'N/A')}

Sandbox Execution Telemetry:
- Status: {sandbox_status}
- Triton Kernels Found: {audit.get('has_triton_kernels', False)}
- ROCm / HIP Found: {audit.get('has_rocm_hip', False)}
- CUDA Found: {audit.get('has_cuda', False)}
- AST Valid: {sandbox_res.get('syntax', {}).get('valid', True)}

README Excerpt:
{readme_snippet}

Evaluate and return ONLY a JSON object with these exact keys:
{{
  "viability_score": <integer 0-100>,
  "alignment_score": <integer 0-100>,
  "verdict": "<MUST_READ | WORTH_EXPLORING | MONITOR | PASS>",
  "executive_pitch": "<1-2 sentences summarizing core innovation>",
  "technical_critique": "<2-3 sentences analyzing code quality, dependencies, and sandbox results>",
  "synergy_notes": "<1-2 actionable ideas for batteryphil / PRIME-Moment-Attention>"
}}
<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        new_tokens = output_tokens[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Parse JSON
        parsed = self._extract_json(generated_text)
        if parsed:
            # Clamp scores to 0-100
            parsed["viability_score"] = max(0, min(100, int(parsed.get("viability_score", 60))))
            parsed["alignment_score"] = max(0, min(100, int(parsed.get("alignment_score", 60))))
            parsed["sandbox_status"] = sandbox_status
            parsed["sandbox_log"] = sandbox_log
            return parsed
            
        print("[!] Failed to parse JSON from LLM output. Using fallback parser.")
        return self._heuristic_evaluate(repo, readme_text, sandbox_res, raw_llm=generated_text)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        # Look for ```json ... ``` or raw {...}
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
                
        match2 = re.search(r"(\{.*\})", text, re.DOTALL)
        if match2:
            try:
                return json.loads(match2.group(1))
            except Exception:
                pass
                
        return None

    def _heuristic_evaluate(self, repo: Dict[str, Any], readme_text: str, sandbox_res: Dict[str, Any], raw_llm: str = "") -> Dict[str, Any]:
        """
        High-quality heuristic evaluator based on keyword scoring, sandbox results, and star telemetry.
        """
        audit = sandbox_res.get("audit", {})
        sandbox_status = sandbox_res.get("status", "SKIPPED")
        
        # 1. Calculate Alignment Score
        full_text = f"{repo.get('name', '')} {repo.get('description', '')} {' '.join(repo.get('topics', []))} {readme_text[:3000]}".lower()
        alignment = 25.0
        
        keywords = {
            "linear attention": 20,
            "prime": 25,
            "moment": 15,
            "triton": 18,
            "rocm": 20,
            "hip": 15,
            "sub-quadratic": 15,
            "recurrent": 12,
            "mamba": 10,
            "kv cache": 15,
            "decision transformer": 15,
            "continuous control": 12,
            "long context": 10
        }
        for kw, boost in keywords.items():
            if kw in full_text:
                alignment += boost
                
        if audit.get("has_triton_kernels") or audit.get("has_rocm_hip"):
            alignment += 15
            
        alignment = min(98, max(20, int(alignment)))
        
        # 2. Calculate Viability Score
        viability = 50.0
        if sandbox_status == "PASS":
            viability += 35
        elif sandbox_status == "MISSING_DEPS":
            viability += 20  # Valid syntax, just needed pip install
        elif sandbox_status == "NO_PACKAGE":
            viability += 15
        elif sandbox_status == "SYNTAX_ERROR":
            viability -= 30
            
        stars = repo.get("stars", 0)
        if stars > 1000:
            viability += 10
        elif stars > 100:
            viability += 5
            
        viability = min(98, max(15, int(viability)))
        
        # 3. Verdict
        if alignment >= 80 and viability >= 70:
            verdict = "MUST_READ"
        elif alignment >= 60 or viability >= 80:
            verdict = "WORTH_EXPLORING"
        elif alignment >= 40:
            verdict = "MONITOR"
        else:
            verdict = "PASS"
            
        pitch = repo.get("description") or f"A research repository focusing on {repo.get('language', 'machine learning')} architectures."
        critique = f"Code AST syntax verified ({sandbox_res.get('syntax', {}).get('files_checked', 0)} files). Hardware detection: Triton={audit.get('has_triton_kernels')}, ROCm={audit.get('has_rocm_hip')}. Sandbox status: {sandbox_status}."
        synergy = f"Directly relevant to linear attention & kernel acceleration. Consider testing their memory tiling or continuous causal formulations against PRIME."

        return {
            "viability_score": viability,
            "alignment_score": alignment,
            "verdict": verdict,
            "executive_pitch": pitch,
            "technical_critique": critique,
            "synergy_notes": synergy,
            "sandbox_status": sandbox_status,
            "sandbox_log": sandbox_res.get("log", "N/A")
        }

if __name__ == "__main__":
    evaluator = PrimeScoutEvaluator(mode="heuristic")
    sample_repo = {
        "full_name": "sustcsonglin/flash-linear-attention",
        "name": "flash-linear-attention",
        "repo_url": "https://github.com/sustcsonglin/flash-linear-attention",
        "stars": 2450,
        "description": "Hardware-accelerated linear attention mechanisms in Triton.",
        "topics": ["linear-attention", "triton", "rocm"]
    }
    sample_sandbox = {
        "status": "PASS",
        "audit": {"has_triton_kernels": True, "has_rocm_hip": True},
        "syntax": {"valid": True, "files_checked": 24},
        "log": "All tests passed."
    }
    res = evaluator.evaluate(sample_repo, "Flash linear attention implementations in Triton.", sample_sandbox)
    print("Evaluator test result:")
    print(json.dumps(res, indent=2))
