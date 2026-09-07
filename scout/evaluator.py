"""
PRIME-Scout: Local Repository Evaluator & Conversational Agent
Evaluates candidate repositories using local LLM (Qwen2.5-Coder-1.5B on ROCm/GPU)
grounded in sandbox telemetry, AST audits, and research alignment.
Supports two-way dialogue between Phil and PRIME-Scout.
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional, List
from scout.config import MODEL_CONFIG, RESEARCH_PROFILE
from scout.database import (
    save_chat_message, get_chat_history,
    save_agent_question, get_pending_questions,
    get_recent_evaluations
)

class PrimeScoutEvaluator:
    def __init__(self, mode: str = "auto", device: Optional[str] = None):
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
        arch = sandbox_res.get("arch_profile", {})
        
        readme_snippet = readme_text[:4000].strip() if readme_text else "No README available."
        
        prompt = f"""<|im_start|>system
You are PRIME-Scout, an elite AI research assistant evaluating repositories for batteryphil, author of PRIME-Moment-Attention (a constant-memory 2nd-order moment attention mechanism on ROCm/AMD GPUs).
Evaluate the candidate repository for research synergy, code viability, architectural innovation, and generate a strategic question for Phil.
SAFETY: You must never commit or push to any git repository.
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
- Layer Classes: {arch.get('found_layer_classes', [])}

README Excerpt:
{readme_snippet}

Evaluate and return ONLY a JSON object with these exact keys:
{{
  "viability_score": <integer 0-100>,
  "alignment_score": <integer 0-100>,
  "verdict": "<MUST_READ | WORTH_EXPLORING | MONITOR | PASS>",
  "executive_pitch": "<1-2 sentences summarizing core innovation>",
  "technical_critique": "<2-3 sentences analyzing code quality, dependencies, and sandbox results>",
  "synergy_notes": "<1-2 actionable ideas for batteryphil / PRIME-Moment-Attention>",
  "question_for_phil": "<1 direct, insightful question for Phil about whether/how to adapt this into PRIME>"
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
        
        parsed = self._extract_json(generated_text)
        if parsed:
            parsed["viability_score"] = max(0, min(100, int(parsed.get("viability_score", 60))))
            parsed["alignment_score"] = max(0, min(100, int(parsed.get("alignment_score", 60))))
            parsed["sandbox_status"] = sandbox_status
            parsed["sandbox_log"] = sandbox_log
            
            # Save question for Phil if present
            q_text = parsed.get("question_for_phil")
            if q_text and repo.get("id"):
                save_agent_question(repo.get("id"), repo.get("name", "Candidate"), q_text)
                
            return parsed
            
        print("[!] Failed to parse JSON from LLM output. Using fallback parser.")
        return self._heuristic_evaluate(repo, readme_text, sandbox_res, raw_llm=generated_text)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
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
        audit = sandbox_res.get("audit", {})
        sandbox_status = sandbox_res.get("status", "SKIPPED")
        
        full_text = f"{repo.get('name', '')} {repo.get('description', '')} {' '.join(repo.get('topics', []))} {readme_text[:3000]}".lower()
        alignment = 25.0
        
        keywords = {
            "linear attention": 20, "prime": 25, "moment": 15, "triton": 18,
            "rocm": 20, "hip": 15, "sub-quadratic": 15, "recurrent": 12,
            "mamba": 10, "kv cache": 15, "decision transformer": 15,
            "continuous control": 12, "long context": 10
        }
        for kw, boost in keywords.items():
            if kw in full_text:
                alignment += boost
                
        if audit.get("has_triton_kernels") or audit.get("has_rocm_hip"):
            alignment += 15
            
        alignment = min(98, max(20, int(alignment)))
        
        viability = 50.0
        if sandbox_status == "PASS":
            viability += 35
        elif sandbox_status == "MISSING_DEPS":
            viability += 20
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
        question = f"Should we implement a micro-benchmark comparing this repo's recurrent forward pass with our Stage 7 Hybrid chunked prefill?"

        if repo.get("id"):
            save_agent_question(repo.get("id"), repo.get("name", "Candidate"), question)

        return {
            "viability_score": viability,
            "alignment_score": alignment,
            "verdict": verdict,
            "executive_pitch": pitch,
            "technical_critique": critique,
            "synergy_notes": synergy,
            "question_for_phil": question,
            "sandbox_status": sandbox_status,
            "sandbox_log": sandbox_res.get("log", "N/A")
        }

    # ==========================================================================
    # TWO-WAY CONVERSATION INTERFACE
    # ==========================================================================
    def chat(self, user_message: str, context_repo_url: Optional[str] = None) -> str:
        """
        Interactive dialogue between Phil and PRIME-Scout.
        Grounded in the repository vault, sandbox experiments, and PRIME thesis.
        """
        save_chat_message(sender="user", content=user_message)

        recent_evals = get_recent_evaluations(limit=5)
        vault_summary = "\n".join([
            f"- {e['full_name']} ({e['verdict']}): Viability {e['viability_score']}/100, Alignment {e['alignment_score']}/100. Pitch: {e['executive_pitch'][:100]}..."
            for e in recent_evals
        ])

        system_prompt = f"""You are PRIME-Scout, an autonomous local AI research partner working directly with Phil (author of batteryphil/PRIME-Moment-Attention).
You specialize in sub-quadratic attention, 2nd-order moment invariants (S0, S1, S2), ROCm/HIP kernels, chunked prefill, Decision Transformers, and generative video.
You have access to candidate repositories cloned into your sandbox (e.g. flash-linear-attention, mamba, recurrent-memory-transformer).
SAFETY RULE: You NEVER make git commits or push code. All experimentation is done in isolated sandbox chambers.

Recent Knowledge Vault:
{vault_summary}

Respond directly, concisely, and technically to Phil's message."""

        if self.mode == "heuristic" or not self._initialized:
            try:
                self.load_model()
            except Exception as e:
                reply = f"[PRIME-Scout] (Offline Fallback) I hear you, Phil. Regarding your question '{user_message}': I'm currently tracking {len(recent_evals)} candidate repositories in the vault with zero-commit sandbox isolation. You can ask me to run benchmarks or evaluate specific URLs anytime."
                save_chat_message(sender="scout", content=reply)
                return reply

        import torch
        prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = output_tokens[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        save_chat_message(sender="scout", content=response)
        return response

if __name__ == "__main__":
    evaluator = PrimeScoutEvaluator(mode="heuristic")
    reply = evaluator.chat("What do you think about combining Mamba with our PRIME second-order accumulator?")
    print("Chat Reply:", reply)
