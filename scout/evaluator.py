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
    save_agent_question, get_pending_questions, get_answered_questions,
    get_recent_evaluations, get_recent_theories
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
        
        # Incorporate continuous learning from Phil's previous answers
        answered = get_answered_questions(limit=5)
        phil_guidance = "\n".join([
            f"- Prior Guidance on {q.get('repo_name', 'Repo')}: \"{q.get('question')}\" -> Phil's Answer: \"{q.get('user_answer')}\""
            for q in answered if q.get("user_answer")
        ]) if answered else "No specific past answers recorded yet."

        prompt = f"""<|im_start|>system
You are PRIME-Scout, an elite AI research assistant evaluating repositories for batteryphil, author of PRIME-Moment-Attention (a constant-memory 2nd-order moment attention mechanism on ROCm/AMD GPUs).
Evaluate the candidate repository for research synergy, code viability, architectural innovation, and generate a strategic question for Phil.
ALIGNMENT WITH PHIL'S PRIOR FEEDBACK:
{phil_guidance}
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
    # TWO-WAY CONVERSATION & INITIATIVE ENGINE
    # ==========================================================================
    def chat(self, user_message: str, context_repo_url: Optional[str] = None) -> str:
        """
        Interactive dialogue between Phil and PRIME-Scout.
        Takes initiative: auto-detects GitHub URLs in Phil's message, clones them
        into the sandbox, audits AST & hardware kernels, and reports live findings.
        Grounded in Phil's foundational research profile (PRIME-Net, Titan MIMO, Thalamic Bloom, PRIME-Moment-Attention).
        """
        save_chat_message(sender="user", content=user_message)

        # 1. Check for Phil's foundational research profile
        from scout.config import VAULT_DIR
        profile_file = VAULT_DIR / "user_profile.json"
        profile_context = ""
        if profile_file.exists():
            try:
                profile_context = profile_file.read_text(errors="replace")
            except Exception:
                pass

        # 2. INITIATIVE: Check if Phil provided or requested a GitHub repository to study
        detected_urls = re.findall(r'https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', user_message)
        initiative_findings = []

        if detected_urls:
            from scout.sandbox import SandboxRunner
            from scout.github_client import GitHubClient
            sandbox = SandboxRunner()
            client = GitHubClient()
            for url in detected_urls[:3]:
                clean_url = url.rstrip('.git').rstrip('/')
                parts = clean_url.split('/')
                owner, name = parts[-2], parts[-1]
                print(f"[*] PRIME-Scout Initiative: Auto-cloning & studying {owner}/{name} from chat...")
                s_res = sandbox.evaluate_repo_sandbox(clean_url, owner, name)
                readme = client.fetch_readme(f"{owner}/{name}")
                arch = s_res.get("arch_profile", {})
                initiative_findings.append(
                    f"Auto-Analyzed [{owner}/{name}]: Status={s_res['status']}. "
                    f"AST Files={s_res.get('syntax', {}).get('files_checked', 0)}. "
                    f"Layers={arch.get('found_layer_classes', [])[:4]}. "
                    f"Triton={s_res.get('audit', {}).get('has_triton_kernels', False)}, "
                    f"ROCm={s_res.get('audit', {}).get('has_rocm_hip', False)}."
                )

        # 3. Compile Vault Context & Human Feedback
        recent_evals = get_recent_evaluations(limit=5)
        vault_summary = "\n".join([
            f"- {e['full_name']} ({e['verdict']}): Viability {e['viability_score']}/100, Alignment {e['alignment_score']}/100."
            for e in recent_evals
        ])
        answered_q = get_answered_questions(limit=5)
        phil_prior_feedback = "\n".join([
            f"- Prior Phil Guidance on {q.get('repo_name', 'Repo')}: \"{q.get('user_answer')}\""
            for q in answered_q if q.get("user_answer")
        ]) if answered_q else "No prior direct feedback recorded yet."

        recent_theories = get_recent_theories(limit=5)
        theories_summary = "\n".join([
            f"- {th['title']} [{th['status']}]: {th['hypothesis']} (Finding: {th['empirical_conclusion']})"
            for th in recent_theories
        ]) if recent_theories else "No formal theories formulated yet."

        system_prompt = f"""You are PRIME-Scout, an elite AI research partner working directly with Phil (batteryphil).
PHIL'S FOUNDATIONAL RESEARCH ECOSYSTEM:
1. PRIME-Net: Pareto-Refined Invariant Mining Engine (Symbolic Regression, Age-Fitness Pareto Optimization, empirical equation synthesis).
2. Titan MIMO PRIME (prime-revisited): 650M-parameter Mamba language model with discrete vote-based optimizer and autonomous MoE routing.
3. Thalamic Bloom (thalamic-bloom): Mamba3 Titan 2.54B with 16 parallel MIMO reasoning arms, sparse IPC 64-dim Blackboard, and thalamic primer.
4. PRIME-Moment-Attention: 2nd-order scalar moment recurrence (S0, S1, S2), Gumbel-Softmax STE discrete routing, Stage 7 Hybrid (25% boundary Softmax, 75% interior PRIME trunk), chunked linear prefill C=256.

PHIL'S PRIOR GUIDANCE & ANSWERS:
{phil_prior_feedback}

AUTONOMOUS THEORIES & EXPERIMENTAL DISCOVERIES:
{theories_summary}

SAFETY: Never commit or push to any git repository.
ACTION INITIATIVE FINDINGS:
{chr(10).join(initiative_findings) if initiative_findings else "No new URLs submitted in this turn."}

Recent Knowledge Vault:
{vault_summary}

Respond directly, technically, and insightfully to Phil as his autonomous research peer."""

        if self.mode == "heuristic" or not self._initialized:
            try:
                self.load_model()
            except Exception as e:
                reply = (
                    f"Phil, I've ingested your message. "
                    f"{' I have cloned and analyzed: ' + '; '.join(initiative_findings) if initiative_findings else ''} "
                    f"I am actively tracking your research across PRIME-Net, Thalamic Bloom (16 MIMO arms / 64-dim Blackboard), and PRIME-Moment-Attention."
                )
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
                max_new_tokens=450,
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
