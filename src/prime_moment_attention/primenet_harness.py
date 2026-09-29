#!/usr/bin/env python3
"""
PRIME-Net Neuro-Symbolic Math Harness
=====================================
A model-agnostic symbolic add-on harness that intercepts mathematical expressions,
evaluates exact arithmetic/algebra using SymPy, and injects verified ground truths
into the model's autoregressive reasoning stream.
"""

import os
import re
import sys
import math
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import sympy as sp
import torch


class PrimeNetMathHarness:
    """
    Plug-and-play symbolic math co-thinker harness for causal language models.
    Supports GSM8K <<expr>> scratchpads, [calc: expr] bracket calls, equation solving,
    and automatic invariant extraction from prompt statements.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._eval_cache: Dict[str, str] = {}
        self.seen_interventions = set()

        # Regex patterns for calculation interception
        self.gsm8k_calc_pattern = re.compile(r"<<([^>=]+)(?:=([^>]*))?>>")
        self.bracket_calc_pattern = re.compile(r"\[(?:calc|eval|primenet):\s*([^\]]+)\]", re.IGNORECASE)
        self.equation_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*([\$]?\s*\d+(?:\.\d+)?)?')

    def safe_eval(self, expr_str: str) -> Optional[Union[int, float, str]]:
        """
        Safely evaluates arithmetic or algebraic expressions using SymPy in < 0.001s.
        """
        raw_str = expr_str.strip()
        if not raw_str:
            return None

        # 1. Algebraic Equation Solving: "Solve for x: 2 * x + 5 = 15" or "2*x + 5 = 15"
        if "solve for" in raw_str.lower() or ("=" in raw_str and re.search(r'\b[a-zA-Z]\b', raw_str)):
            eq_text = re.sub(r'solve\s+(?:for\s+)?([a-zA-Z])\s*[:\s]*', '', raw_str, flags=re.I).strip().rstrip('?.')
            if "=" in eq_text:
                lhs_str, rhs_str = eq_text.split("=", 1)
                vars_found = list(set(re.findall(r'\b([a-zA-Z])\b', eq_text)))
                var_sym = sp.Symbol(vars_found[0]) if vars_found else sp.Symbol('x')
                try:
                    lhs = sp.sympify(lhs_str.replace('^', '**'))
                    rhs = sp.sympify(rhs_str.replace('^', '**'))
                    sol = sp.solve(sp.Eq(lhs, rhs), var_sym)
                    if sol:
                        val = sol[0]
                        if val.is_number:
                            return int(val) if val == int(val) else float(val)
                        return str(val)
                except Exception:
                    pass

        # 2. Percentage Evaluation: "20 percent of 150" or "20% of 150"
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)?\s*(\d+(?:\.\d+)?)', raw_str, re.IGNORECASE)
        if pct_match:
            p_val = float(pct_match.group(1))
            base_val = float(pct_match.group(2))
            res = (p_val / 100.0) * base_val
            return int(res) if res == int(res) else res

        # 3. Clean standard arithmetic
        clean_expr = raw_str
        clean_expr = re.sub(r'[=\?:\$]', '', clean_expr).strip()
        clean_expr = clean_expr.replace('×', '*').replace('÷', '/').replace('^', '**')
        clean_expr = re.sub(r'(\d+),(\d+)', r'\1\2', clean_expr)

        # Basic security sanitization
        if not re.match(r"^[0-9a-zA-Z_\+\-\*\/\%\.\(\)\,\s\*\*\*]+$", clean_expr):
            return None

        if clean_expr in self._eval_cache:
            return self._eval_cache[clean_expr]

        try:
            val = sp.sympify(clean_expr, rational=True)
            if val.is_integer:
                res = int(val)
            elif val.is_rational:
                f_val = float(val)
                res = int(f_val) if f_val.is_integer() else round(f_val, 4)
            elif val.is_number:
                f_val = float(val.evalf())
                res = int(f_val) if f_val.is_integer() else round(f_val, 4)
            else:
                res = str(sp.simplify(val))

            self._eval_cache[clean_expr] = res
            return res
        except Exception:
            return None

    def extract_prompt_invariants(self, prompt: str) -> Optional[str]:
        """
        Extracts mathematical relationships, sequence rules, equations, or rate problems
        from prompt questions to prime the generation context with verified ground truths.
        """
        # 1. Algebraic Equation: "Solve for x: 3 * x + 7 = 22"
        alg_match = re.search(r'(?:solve\s+(?:for\s+[a-zA-Z]\s*[:\s]*)?|find\s+[a-zA-Z]\s*[:\s]*)([^,\n\?]+=[^,\n\?\.]+)', prompt, re.IGNORECASE)
        if alg_match:
            eq_str = alg_match.group(1).strip()
            sol = self.safe_eval(eq_str)
            if sol is not None:
                return f"[PRIME-Net Invariant: {eq_str} -> x = {sol}]"

        # 2. Percentage Question: "What is 25 percent of 200?"
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)\s*(\d+(?:\.\d+)?)', prompt, re.IGNORECASE)
        if pct_match:
            p_val = float(pct_match.group(1))
            base_val = float(pct_match.group(2))
            res = (p_val / 100.0) * base_val
            res_str = f"{int(res)}" if res == int(res) else f"{res}"
            return f"[PRIME-Net Invariant: {pct_match.group(1)}% of {pct_match.group(2)} = {res_str}]"

        # 3. Direct calculation prompt: "Calculate 25 * 16" or "What is 80000 + 50000"
        # Guard: Do not extract partial arithmetic if exponents (^) or modular expressions are present
        if "^" not in prompt and "**" not in prompt and "remainder" not in prompt.lower() and "mod" not in prompt.lower():
            calc_match = re.search(r'(?:calculate|what is|compute|find)\s+([\d\.\,\s\+\-\*\/\(\)]+[\+\-\*\/][\d\.\,\s\+\-\*\/\(\)]+)', prompt, re.IGNORECASE)
            if not calc_match:
                calc_match = re.search(r'(\b\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)*\b)', prompt)
            if calc_match:
                expr = calc_match.group(1).strip().rstrip('?.')
                res = self.safe_eval(expr)
                if res is not None:
                    return f"[PRIME-Net Invariant: {expr} = {res}]"

        # 4. Multi-Item Unit Pricing: "3 books for 15 dollars each and 2 pens for 4 dollars each"
        unit_matches = re.findall(r"(\d+(?:\.\d+)?)\s+[a-zA-Z]+\s+(?:for|at)\s+\$?(\d+(?:\.\d+)?)\s*(?:dollars?|cents?)?\s*each", prompt, re.I)
        if len(unit_matches) >= 2:
            sub_exprs = [f"({int(float(m[0])) if float(m[0]) == int(float(m[0])) else m[0]} * {int(float(m[1])) if float(m[1]) == int(float(m[1])) else m[1]})" for m in unit_matches]
            expr = " + ".join(sub_exprs)
            total = sum(float(m[0]) * float(m[1]) for m in unit_matches)
            total_val = int(total) if total == int(total) else total
            return f"[PRIME-Net Invariant: {expr} = {total_val}]"

        # 5. Word Problem Total Cost / Spending
        if re.search(r'\b(in total|total cost|total spent|spend in total|altogether|combined|sum of)\b', prompt, re.IGNORECASE):
            nums = [float(x.replace(",", "")) for x in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", prompt)]
            if len(nums) >= 2 and not re.search(r'\b(each|per)\b', prompt, re.I):
                expr = " + ".join(str(int(x) if x == int(x) else x) for x in nums)
                total_val = sum(nums)
                res = int(total_val) if total_val == int(total_val) else total_val
                return f"[PRIME-Net Invariant: {expr} = {res}]"

        # 6. Word Problem Remaining / Left
        if re.search(r'\b(left|remaining|remains|leftover)\b', prompt, re.IGNORECASE):
            nums = [float(x.replace(",", "")) for x in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", prompt)]
            if len(nums) >= 2 and not re.search(r'\b(each|per)\b', prompt, re.I):
                expr = f"{int(nums[0]) if nums[0] == int(nums[0]) else nums[0]} - " + " - ".join(str(int(x) if x == int(x) else x) for x in nums[1:])
                left_val = nums[0] - sum(nums[1:])
                res = int(left_val) if left_val == int(left_val) else left_val
                return f"[PRIME-Net Invariant: {expr} = {res}]"

        # 7. Sequence rule discovery (polynomial, geometric)
        seq_match = re.search(r'((?:[-+]?\d+(?:\.\d+)?,\s*){3,}[-+]?\d+(?:\.\d+)?)', prompt)
        if seq_match:
            try:
                nums = [float(x.strip()) for x in seq_match.group(1).split(',') if x.strip()]
                if len(nums) >= 3:
                    n = len(nums)
                    x = np.arange(1, n + 1)
                    y = np.array(nums)
                    for deg in [1, 2, 3]:
                        coeffs = np.polyfit(x, y, deg)
                        p = np.poly1d(coeffs)
                        if np.allclose(p(x), y, atol=1e-4):
                            terms = []
                            for i, c in enumerate(coeffs):
                                power = deg - i
                                c_val = round(c, 4)
                                if abs(c_val) < 1e-4:
                                    continue
                                c_str = f"{int(c_val)}" if c_val == int(c_val) else f"{c_val}"
                                if power == 0:
                                    terms.append(f"{c_str}")
                                elif power == 1:
                                    terms.append(f"n" if c_str == "1" else f"{c_str}*n")
                                else:
                                    terms.append(f"n**{power}" if c_str == "1" else f"{c_str}*n**{power}")
                            rule = " + ".join(terms).replace("+ -", "- ")
                            return f"[PRIME-Net Invariant: f(n) = {rule}]"
            except Exception:
                pass

        return None

    def intercept_and_solve(self, text_chunk: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scans generated text, evaluates calculations, and returns updated text with verified results.
        """
        injections = []
        seen = set()

        # 1. GSM8K format: <<expr>> or <<expr=wrong>>
        def gsm8k_repl(match):
            expr = match.group(1).strip()
            if expr.lower() in ("expression", "formula", "calc", "example"):
                return match.group(0)
            res = self.safe_eval(expr)
            if res is not None:
                if expr not in seen:
                    seen.add(expr)
                    injections.append({"expr": expr, "result": str(res), "type": "gsm8k"})
                return f"<<{expr}={res}>>{res}"
            return match.group(0)

        updated_text = self.gsm8k_calc_pattern.sub(gsm8k_repl, text_chunk)

        # 2. Bracket format: [calc: 25 * 14]
        def bracket_repl(match):
            expr = match.group(1).strip()
            res = self.safe_eval(expr)
            if res is not None:
                if expr not in seen:
                    seen.add(expr)
                    injections.append({"expr": expr, "result": str(res), "type": "bracket"})
                return f"[PRIME-Net: {expr} = {res}]"
            return match.group(0)

        updated_text = self.bracket_calc_pattern.sub(bracket_repl, updated_text)

        # 3. Equality format: X op Y =
        m = self.equation_pattern.search(updated_text)
        if m:
            n1_str, op, n2_str, res_str = m.groups()
            expr = f"{n1_str} {op} {n2_str}"
            val = self.safe_eval(expr)
            if val is not None and (res_str is None or not res_str.strip()):
                if expr not in seen:
                    seen.add(expr)
                    injections.append({"expr": expr, "result": str(val), "type": "equality"})
                updated_text = updated_text[:m.end()].rstrip() + f" {val} "

        return updated_text, injections

    def generate_with_harness(
        self,
        model: torch.nn.Module,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.2,
        top_k: int = 20,
        device: Optional[str] = None,
    ) -> str:
        """
        Model-agnostic generation function: streams tokens, intercepts calculation tags,
        and dynamically injects exact SymPy results into the generation context.
        """
        if device is None:
            device = next(model.parameters()).device

        # 1. Check prompt invariant
        inv = self.extract_prompt_invariants(prompt)
        augmented_prompt = prompt
        if inv and inv not in augmented_prompt:
            if "<think>" in augmented_prompt:
                parts = augmented_prompt.split("<think>", 1)
                augmented_prompt = parts[0] + "<think>\n" + inv + "\n" + parts[1].lstrip()
            else:
                augmented_prompt = augmented_prompt.rstrip() + f"\n{inv}\nAnswer:"

        input_ids = tokenizer.encode(augmented_prompt, return_tensors="pt").to(device)
        generated_ids = input_ids[0].tolist()

        buf = ""
        model.eval()

        for step in range(max_new_tokens):
            inp = torch.tensor([generated_ids], device=device)
            with torch.no_grad():
                out = model(inp)
                logits = out.logits[:, -1, :]

            if temperature > 0.0:
                scaled = logits / max(temperature, 1e-4)
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits.argmax(dim=-1).item()

            generated_ids.append(next_token)
            tok_str = tokenizer.decode([next_token])
            buf += tok_str

            # Check if an arithmetic trigger appeared
            if ">>" in buf or "]" in buf or "=" in buf:
                recent_text = tokenizer.decode(generated_ids[-30:])
                updated_recent, injections = self.intercept_and_solve(recent_text)
                if injections:
                    # Re-encode the updated text snippet
                    prefix_ids = generated_ids[:-30]
                    new_recent_ids = tokenizer.encode(updated_recent, add_special_tokens=False)
                    generated_ids = prefix_ids + new_recent_ids
                    buf = ""
                    if self.verbose:
                        print(f"  [PRIME-Net Math Injected]: {injections}")

            if next_token == tokenizer.eos_token_id:
                break

            # If #### appeared with answer, stop
            if "####" in tokenizer.decode(generated_ids[-10:]):
                ans_str = tokenizer.decode(generated_ids).split("####")[-1]
                if any(c.isdigit() for c in ans_str) and ("\n" in ans_str or len(ans_str) > 8):
                    break

        return tokenizer.decode(generated_ids, skip_special_tokens=True)
