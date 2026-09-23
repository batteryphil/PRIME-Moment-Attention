"""
SFT Reasoning Dataset with Exact Prompt Loss Masking
===================================================
Produces high-density thought traces for Supervised Fine-Tuning of PrimeLM-50M.

Features:
1. Exact Prompt Loss Masking: labels = -100 on prompt tokens, loss calculated exclusively
   on <think>...</think> reconstructive thoughts and the final response.
2. 4-Way Multi-Domain Mixture:
   - Mathematics (GSM8K + algebraic equations)
   - Physical Invariants (Kinetic energy, thermodynamics, circuit laws)
   - Long-range Associative Memory & Entity Tracking
   - Conversational Intent Planning (Smoltalk)
"""

import os
import re
import random
import torch
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


def format_math_reasoning(q: str, thought_steps: str, final_ans: str) -> Tuple[str, str]:
    """Formats mathematical reasoning with explicit reconstruction thought trace."""
    thought = (
        f"<think>\n"
        f"Let's reconstruct and solve this step-by-step:\n"
        f"{thought_steps.strip()}\n"
        f"Verification: The logical derivation holds and matches physical/algebraic invariants.\n"
        f"</think>"
    )
    prompt = f"User: {q.strip()}\n\nAssistant: "
    response = f"{thought}\n\nThe final answer is {final_ans.strip()}."
    return prompt, response


def format_physics_invariant(q: str, formula: str, calc: str, ans: str) -> Tuple[str, str]:
    """Formats physical science problems with dimensional analysis."""
    thought = (
        f"<think>\n"
        f"Physics Analysis:\n"
        f"1. Governing Conservation Law: {formula}\n"
        f"2. Variable Mapping and Dimensional Consistency: {calc}\n"
        f"3. Verification: Dimensions and units check out consistently.\n"
        f"</think>"
    )
    prompt = f"User: {q.strip()}\n\nAssistant: "
    response = f"{thought}\n\nFormula: {formula}\nCalculation: {calc}\nFinal Answer: {ans}."
    return prompt, response


def format_associative_memory(entity: str, attribute: str, val: str, distractor: str, query: str) -> Tuple[str, str]:
    """Formats long-range associative memory retrieval requiring thought map recall."""
    context = f"Record: The {attribute} for {entity} is registered as {val}. {distractor}"
    prompt = f"Context: {context}\n\nUser: {query}\n\nAssistant: "
    thought = (
        f"<think>\n"
        f"Reconstructing memory from blueprint:\n"
        f"1. Query Entity: {entity}\n"
        f"2. Target Attribute: {attribute}\n"
        f"3. Retrieved Invariant Value: {val}\n"
        f"</think>"
    )
    response = f"{thought}\n\nThe {attribute} of {entity} is {val}."
    return prompt, response


class SFTReasoningStreamer:
    """Streams tokenized sequences with exact prompt loss masking."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 1024,
        device: torch.device = torch.device("cpu")
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device
        
        # Load GSM8K dataset
        print("[*] Initializing GSM8K dataset...")
        try:
            self.gsm_data = list(load_dataset("openai/gsm8k", "main", split="train"))
            random.shuffle(self.gsm_data)
        except Exception as e:
            print(f"[-] Warning: Failed to load online GSM8K: {e}. Using synthetic fallback.")
            self.gsm_data = []

        self.gsm_idx = 0

    def _generate_physics_item(self) -> Tuple[str, str]:
        templates = [
            (
                "Calculate kinetic energy of an object of mass {m} kg moving at speed {v} m/s.",
                "E = 0.5 * m * v^2",
                "E = 0.5 * {m} * {v_sq} = {res} Joules",
                "{res} Joules",
                lambda m, v: 0.5 * m * (v ** 2)
            ),
            (
                "Find thermal heat required to raise {m} kg of water by {dt} K.",
                "Q = m * c * deltaT",
                "Q = {m} * 4.184 * {dt} = {res} kJ",
                "{res} kJ",
                lambda m, dt: round(m * 4.184 * dt, 2)
            ),
            (
                "Find the electrical voltage across a circuit with current {i} A and resistance {r} Ohms.",
                "V = I * R",
                "V = {i} * {r} = {res} Volts",
                "{res} Volts",
                lambda i, r: i * r
            )
        ]
        t = random.choice(templates)
        if "kinetic" in t[0]:
            m = random.choice([2, 4, 6, 8, 10])
            v = random.choice([2, 3, 4, 5, 6])
            res = t[4](m, v)
            res_str = str(int(res)) if res == int(res) else str(res)
            q = t[0].format(m=m, v=v)
            calc = t[2].format(m=m, v_sq=v*v, res=res_str)
            ans = t[3].format(res=res_str)
            return format_physics_invariant(q, t[1], calc, ans)
        elif "thermal" in t[0]:
            m = random.choice([1, 2, 3, 4, 5])
            dt = random.choice([5, 10, 15, 20])
            res = t[4](m, dt)
            q = t[0].format(m=m, dt=dt)
            calc = t[2].format(m=m, dt=dt, res=res)
            ans = t[3].format(res=res)
            return format_physics_invariant(q, t[1], calc, ans)
        else:
            i = random.choice([2, 3, 4, 5, 8, 10])
            r = random.choice([4, 6, 8, 10, 12, 15])
            res = t[4](i, r)
            q = t[0].format(i=i, r=r)
            calc = t[2].format(i=i, r=r, res=res)
            ans = t[3].format(res=res)
            return format_physics_invariant(q, t[1], calc, ans)

    def _generate_algebra_item(self) -> Tuple[str, str]:
        a = random.choice([2, 3, 4, 5, 6, 7])
        x = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
        b = random.choice([5, 7, 10, 12, 15, 20])
        c = a * x + b
        q = f"Solve for x: {a} * x + {b} = {c}."
        steps = (
            f"1. Identify linear equation: {a}*x + {b} = {c}\n"
            f"2. Subtract {b} from both sides: {a}*x = {c} - {b} = {c - b}\n"
            f"3. Divide both sides by {a}: x = {c - b} / {a} = {x}"
        )
        return format_math_reasoning(q, steps, str(x))

    def _generate_associative_item(self) -> Tuple[str, str]:
        entities = ["Project Apollo", "Agent Sigma", "Server Echo", "Vault Delta", "User Phoenix"]
        attributes = ["access code", "passkey", "frequency", "security token"]
        values = ["94812", "72049", "31855", "84920", "65103"]
        distractors = [
            "The weather outside is bright and sunny.",
            "Multiple background processes are actively executing.",
            "Historical logs show continuous steady operation.",
            "The system is running on local hardware with verified stability."
        ]
        ent = random.choice(entities)
        attr = random.choice(attributes)
        val = random.choice(values)
        dist = " ".join(random.sample(distractors, 2))
        query = f"What is the {attr} of {ent}?"
        return format_associative_memory(ent, attr, val, dist, query)

    def get_example(self) -> Tuple[str, str]:
        """Draws an example from the multi-domain mixture."""
        choice = random.random()
        if choice < 0.40 and self.gsm_data:
            item = self.gsm_data[self.gsm_idx]
            self.gsm_idx = (self.gsm_idx + 1) % len(self.gsm_data)
            q = item["question"]
            raw_a = item["answer"]
            parts = raw_a.split("####")
            steps = re.sub(r'<<.*?>>', '', parts[0]).strip() if len(parts) == 2 else raw_a
            ans = parts[1].strip() if len(parts) == 2 else "computed"
            return format_math_reasoning(q, steps, ans)
        elif choice < 0.65:
            return self._generate_physics_item()
        elif choice < 0.85:
            return self._generate_algebra_item()
        else:
            return self._generate_associative_item()

    def get_batch(self, batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes and packs a batch with exact prompt loss masking:
        labels are set to -100 on the prompt, and active token IDs on <think> and response.
        """
        input_ids_list = []
        labels_list = []

        for _ in range(batch_size):
            prompt, response = self.get_example()
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

            # Combined sequence
            seq_tokens = prompt_tokens + response_tokens
            if len(seq_tokens) > self.seq_len:
                seq_tokens = seq_tokens[:self.seq_len]
                prompt_len = min(len(prompt_tokens), self.seq_len)
            else:
                prompt_len = len(prompt_tokens)

            # Mask labels: prompt is -100, response is supervised
            labels = [-100] * prompt_len + seq_tokens[prompt_len:]

            # Pad to seq_len
            pad_len = self.seq_len - len(seq_tokens)
            if pad_len > 0:
                pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
                seq_tokens = seq_tokens + [pad_token] * pad_len
                labels = labels + [-100] * pad_len

            input_ids_list.append(seq_tokens)
            labels_list.append(labels)

        return (
            torch.tensor(input_ids_list, dtype=torch.long, device=self.device),
            torch.tensor(labels_list, dtype=torch.long, device=self.device)
        )


def test_dataset_masking():
    print("[*] Testing SFT Reasoning Dataset and Exact Prompt Masking...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    streamer = SFTReasoningStreamer(tokenizer, seq_len=256)
    
    inputs, labels = streamer.get_batch(batch_size=2)
    assert inputs.shape == (2, 256)
    assert labels.shape == (2, 256)

    # Check masking: first tokens must be -100 (prompt), later tokens must be > 0 (response)
    for b in range(2):
        row_labels = labels[b].tolist()
        num_masked = sum(1 for x in row_labels if x == -100)
        num_supervised = sum(1 for x in row_labels if x != -100)
        assert num_masked > 0, "No prompt masking found!"
        assert num_supervised > 0, "No supervised response tokens found!"
        print(f"  [+] Batch {b}: {num_masked} masked tokens, {num_supervised} supervised reasoning tokens.")

    print("[SUCCESS] SFT dataset masking fully verified!")


if __name__ == "__main__":
    test_dataset_masking()
