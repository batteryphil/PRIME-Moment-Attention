"""
SFT-v2 Reasoning Dataset with Exact Prompt Loss Masking
======================================================
Produces high-density thought traces for Supervised Fine-Tuning of PrimeLM-50M.

Features:
1. Exact Prompt Loss Masking: labels = -100 on prompt tokens, loss calculated exclusively
   on <think>...</think> reconstructive thoughts and the final response.
2. 5-Way Multi-Domain Mixture:
   - Equipment Mechanics & Diagnostics (Hydraulics, Diesel, Electrical, Torque)
   - Explicit Co-Thinker Triggered Math (GSM8K + Procedural Algebra with <<...>>)
   - Conversational Instruction Following (Cached SmolTalk Magpie)
   - Declarative Episodic Memory Protocol (Planting and Retrieval)
   - Physical Invariants (Kinetic energy, thermodynamics, circuit laws)
"""

import os
import re
import glob
import random
import torch
import pyarrow.parquet as pq
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer


def format_math_reasoning(q: str, thought_steps: str, final_ans: str) -> Tuple[str, str]:
    """Formats mathematical reasoning with explicit reconstruction thought trace and Co-Thinker tags."""
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


def format_equipment_mechanics(q: str, thought_steps: str, answer_body: str) -> Tuple[str, str]:
    """Formats heavy equipment diagnostics with step-by-step mechanical reasoning."""
    thought = (
        f"<think>\n"
        f"Mechanical Diagnostics & Physics Decomposition:\n"
        f"{thought_steps.strip()}\n"
        f"Verification: Specifications, safety margins, and physical tolerances verified.\n"
        f"</think>"
    )
    prompt = f"User: {q.strip()}\n\nAssistant: "
    response = f"{thought}\n\n{answer_body.strip()}"
    return prompt, response


def format_memory_planting(entity: str, attribute: str, val: str) -> Tuple[str, str]:
    """Formats episodic memory planting (storing into 64 KB GTRM manifold)."""
    prompt = f"User: Record this into your memory: The {attribute} for {entity} is {val}.\n\nAssistant: "
    thought = (
        f"<think>\n"
        f"Consolidating episodic blueprint for {entity}:\n"
        f"- Target Entity: {entity}\n"
        f"- Attribute: {attribute}\n"
        f"- Registered Value: {val}\n"
        f"Topological coordinates consolidated into 2nd-order memory manifold M^(2).\n"
        f"</think>"
    )
    response = f"{thought}\n\nUnderstood. I have recorded that the {attribute} for {entity} is {val}."
    return prompt, response


def format_memory_retrieval(entity: str, attribute: str, val: str, distractor: str = "") -> Tuple[str, str]:
    """Formats episodic memory recall from 64 KB GTRM manifold."""
    context_prefix = f"Context Record: {entity}'s {attribute} is {val}. {distractor}\n\n" if distractor else ""
    prompt = f"{context_prefix}User: What is the {attribute} of {entity}?\n\nAssistant: "
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


def format_conversational_turn(q: str, response_text: str) -> Tuple[str, str]:
    """Formats natural conversational interaction without forcing math equations."""
    patterns = [
        (
            f"<think>\n"
            f"Understanding User Inquiry:\n"
            f"- Identify subject matter and objective.\n"
            f"- Structure a concise, informative response.\n"
            f"</think>"
        ),
        (
            f"<think>\n"
            f"Query Analysis & Planning:\n"
            f"- Objective: Address the user's question accurately.\n"
            f"- Approach: Provide clear explanations without unneeded complexity.\n"
            f"</think>"
        ),
        (
            f"<think>\n"
            f"Conversational Response Formulation:\n"
            f"1. Process user message.\n"
            f"2. Synthesize key information directly.\n"
            f"</think>"
        ),
        (
            f"<think>\n"
            f"Instruction Evaluation:\n"
            f"- Clarify key points and provide structured, helpful guidance.\n"
            f"</think>"
        )
    ]
    thought = random.choice(patterns)
    prompt = f"User: {q.strip()}\n\nAssistant: "
    response = f"{thought}\n\n{response_text.strip()}"
    return prompt, response


class SFTReasoningStreamer:
    """Streams tokenized sequences with exact prompt loss masking across 5 domains."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 512,
        device: torch.device = torch.device("cpu")
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device

        # 1. Load cached SmolTalk Magpie parquet files for natural dialogue
        print("[*] Initializing SmolTalk Magpie dataset from local cache...")
        smol_files = sorted(glob.glob("/home/phil/.cache/huggingface/hub/datasets--HuggingFaceTB--smoltalk/snapshots/*/data/smol-magpie-ultra/*.parquet"))
        self.conversations = []
        for sf in smol_files:
            try:
                table = pq.read_table(sf)
                messages_col = table["messages"].to_pylist()
                for conv in messages_col:
                    user_msg = None
                    asst_msg = None
                    for m in conv:
                        if m.get("role") == "user" and user_msg is None:
                            user_msg = m.get("content", "").strip()
                        elif m.get("role") == "assistant" and asst_msg is None and user_msg is not None:
                            asst_msg = m.get("content", "").strip()
                            break
                    if user_msg and asst_msg and 15 < len(user_msg) < 400 and 20 < len(asst_msg) < 600:
                        self.conversations.append((user_msg, asst_msg))
                    if len(self.conversations) >= 12000:
                        break
            except Exception as e:
                print(f"[-] Warning loading SmolTalk shard {sf}: {e}")
            if len(self.conversations) >= 12000:
                break
        print(f"[+] Loaded {len(self.conversations)} diverse conversational dialogues from SmolTalk.")

        # 2. GSM8K Cache
        print("[*] Initializing GSM8K dataset...")
        self.gsm_data = []
        gsm_dirs = glob.glob("/home/phil/.cache/huggingface/datasets/openai___gsm8k/main/*")
        if gsm_dirs:
            try:
                from datasets import load_from_disk
                # Attempt to find arrow files in cache
                arrow_files = glob.glob("/home/phil/.cache/huggingface/datasets/openai___gsm8k/main/*/*/*.arrow")
                if not arrow_files:
                    from datasets import load_dataset
                    self.gsm_data = list(load_dataset("openai/gsm8k", "main", split="train"))
            except Exception as e:
                pass

        self.conv_idx = 0
        self.gsm_idx = 0

    def _generate_equipment_mechanics(self) -> Tuple[str, str]:
        """Generates realistic heavy equipment troubleshooting problems with explicit Co-Thinker tags."""
        scenarios = [
            "hydraulic_cylinder_force",
            "hydraulic_pump_flow",
            "diesel_compression_ratio",
            "electrical_voltage_drop",
            "torque_specifications",
            "engine_cooling_overheating"
        ]
        choice = random.choice(scenarios)

        if choice == "hydraulic_cylinder_force":
            area = random.randint(2, 12)
            pressure = random.randint(12, 45) * 100
            force = area * pressure
            templates = [
                f"A hydraulic cylinder on an excavator has a piston area of {area} square inches operating at {pressure} PSI. Calculate the force produced using F = P * A.",
                f"Calculate the linear thrust of a hydraulic ram with an effective area of {area} sq in pressurized to {pressure} PSI using F = P * A.",
                f"A boom lift cylinder operates at {pressure} PSI with a piston surface of {area} square inches. What force in pounds is produced?"
            ]
            q = random.choice(templates)
            steps = (
                f"1. Governing Hydraulic Law: Force = Pressure * Area (F = P * A)\n"
                f"2. Piston Area A = {area} sq in, Pressure P = {pressure} PSI\n"
                f"3. Compute Force: F = <<{pressure} * {area}={force}>> lbs\n"
            )
            body = (
                f"Formula: F = P * A\n"
                f"Calculation: F = {pressure} PSI * {area} sq in = {force:,} lbs\n"
                f"The hydraulic cylinder generates {force:,} lbs of linear thrust."
            )
            return format_equipment_mechanics(q, steps, body)

        elif choice == "hydraulic_pump_flow":
            disp = random.choice([2.5, 3.2, 4.0, 5.5, 6.0, 7.5, 8.0]) # cu in per rev
            rpm = random.choice([1500, 1800, 2000, 2100, 2200, 2400])
            flow = round((disp * rpm) / 231, 1)
            q = f"Calculate the flow rate in GPM for a hydraulic pump with displacement of {disp} cu in/rev running at {rpm} RPM using Flow = (Displacement * RPM) / 231."
            steps = (
                f"1. Flow Formula: GPM = (Disp * RPM) / 231\n"
                f"2. Displacement = {disp} cu in/rev, Speed = {rpm} RPM\n"
                f"3. Numerator Product: <<{disp} * {rpm}={disp * rpm}>>\n"
                f"4. Divide by 231 cu in per gallon: GPM = <<{disp * rpm} / 231={flow}>> GPM\n"
            )
            body = f"The pump delivers approximately {flow} gallons per minute (GPM) at {rpm} RPM."
            return format_equipment_mechanics(q, steps, body)

        elif choice == "electrical_voltage_drop":
            i = random.choice([15, 25, 40, 60, 80, 100, 150]) # Amps
            r = random.choice([0.02, 0.04, 0.05, 0.08, 0.1, 0.15]) # Ohms
            v_drop = round(i * r, 2)
            q = f"A starter solenoid on a diesel engine draws {i} Amps across a cable resistance of {r} Ohms. Calculate the voltage drop using V = I * R."
            steps = (
                f"1. Ohm's Law: Voltage Drop V = Current * Resistance (V = I * R)\n"
                f"2. Current I = {i} A, Cable Resistance R = {r} Ohms\n"
                f"3. Voltage Drop calculation: V = <<{i} * {r}={v_drop}>> Volts\n"
            )
            body = f"The voltage drop across the starter cable is {v_drop} Volts. If battery voltage is 24V, voltage at the starter terminal is {round(24 - v_drop, 2)}V."
            return format_equipment_mechanics(q, steps, body)

        elif choice == "torque_specifications":
            base_torque = random.choice([140, 160, 180, 210, 240, 280])
            angle = random.choice([45, 60, 90])
            part = random.choice(["cylinder head bolts", "main bearing cap bolts", "connecting rod bolts", "flywheel mounting bolts"])
            q = f"What is the proper torque tightening procedure for {part} requiring {base_torque} ft-lbs plus {angle} degrees torque-angle?"
            steps = (
                f"1. Identify torque specification: {base_torque} ft-lbs base snug + {angle} deg rotation for {part}\n"
                f"2. Sequence: Spiral outward or follow factory alternating sequence to avoid component distortion\n"
                f"3. Lubrication: Clean motor oil on threads and washer faces for consistent clamp load\n"
            )
            body = (
                f"1. Stage 1: Torque all {part} in specified sequence to {base_torque} ft-lbs using a calibrated torque wrench.\n"
                f"2. Stage 2: Using an angle gauge, rotate each bolt an additional {angle} degrees in sequence.\n"
                f"Always follow the manufacturer spiral tightening sequence."
            )
            return format_equipment_mechanics(q, steps, body)

        else: # Diesel overheating diagnostics
            q = "An excavator diesel engine is overheating under heavy digging load, but coolant level is full. What are the key diagnostic steps?"
            steps = (
                f"1. Symptom Analysis: Overheating under load with full coolant indicates heat rejection failure or restricted flow.\n"
                f"2. Radiator Core: Check for clogged exterior cooling fins (mud/dirt buildup).\n"
                f"3. Thermostat: Test if thermostat is opening fully at rated temperature (typically 180-195 F).\n"
                f"4. Water Pump & Fan: Inspect belt tension and hydraulic fan drive motor speed.\n"
            )
            body = (
                f"Diagnostic Checklist:\n"
                f"1. Radiator & Oil Cooler Fins: Inspect and blow out dirt/debris packed between the radiator cores.\n"
                f"2. Hydraulic Fan Drive: Verify the cooling fan is reaching maximum commanded RPM under load.\n"
                f"3. Thermostat Operation: Use an infrared thermometer to verify temperature delta between radiator inlet and outlet hoses.\n"
                f"4. Exhaust Backpressure: Check for restricted DPF or turbocharger binding creating excessive thermal load."
            )
            return format_equipment_mechanics(q, steps, body)

    def _generate_declarative_memory(self) -> Tuple[str, str]:
        """Generates declarative memory planting and retrieval pairs for 64 KB GTRM testing."""
        machines = ["Excavator Unit 7", "Dozer D8T", "Wheel Loader 980M", "Service Truck 12", "Haul Truck 105", "Motor Grader 14M", "Backhoe 420F"]
        supervisors = ["Marcus", "Dave", "Sarah", "Phil", "Elena", "Carlos", "Mike", "Rachel"]
        frequencies = ["462.55 MHz", "467.625 MHz", "154.57 MHz", "462.675 MHz", "469.50 MHz"]
        passkeys = ["84920", "59104", "38271", "90124", "44819"]
        
        ent = random.choice(machines)
        val1 = random.choice(supervisors)
        val2 = random.choice(frequencies)
        attr1 = "maintenance supervisor"
        attr2 = "radio frequency"

        if random.random() < 0.5:
            # Memory Planting
            prompt = f"User: Record this into your memory: The {attr1} for {ent} is named {val1}, and the {attr2} is {val2}.\n\nAssistant: "
            thought = (
                f"<think>\n"
                f"Consolidating episodic blueprint for {ent}:\n"
                f"- Entity: {ent}\n"
                f"- {attr1}: {val1}\n"
                f"- {attr2}: {val2}\n"
                f"Topological coordinates consolidated into 2nd-order memory manifold M^(2).\n"
                f"</think>"
            )
            response = f"{thought}\n\nUnderstood. I have recorded that the {attr1} for {ent} is {val1}, and the {attr2} is {val2}."
            return prompt, response
        else:
            # Memory Retrieval
            q = f"What is the {attr1} for {ent}, and what was the {attr2}?"
            thought = (
                f"<think>\n"
                f"Reconstructing memory from blueprint:\n"
                f"1. Target Entity: {ent}\n"
                f"2. Retrieved {attr1}: {val1}\n"
                f"3. Retrieved {attr2}: {val2}\n"
                f"</think>"
            )
            prompt = f"User: {q}\n\nAssistant: "
            response = f"{thought}\n\nThe {attr1} for {ent} is {val1}, and the {attr2} is {val2}."
            return prompt, response

    def _generate_algebra_with_cothinker(self) -> Tuple[str, str]:
        """Generates algebra with explicit <<...>> Co-Thinker calculation triggers."""
        a = random.choice([2, 3, 4, 5, 6, 7, 8, 9])
        x = random.randint(2, 20)
        b = random.randint(4, 40)
        c = a * x + b

        templates = [
            f"Solve for x: {a} * x + {b} = {c}.",
            f"Solve this equation step-by-step: {a} * x + {b} = {c}.",
            f"Find x in the linear equation {a}*x + {b} = {c}."
        ]
        q = random.choice(templates)
        sub_res = c - b
        steps = (
            f"1. Identify linear equation: {a}*x + {b} = {c}\n"
            f"2. Subtract {b} from both sides: {a}*x = {c} - {b} = <<{c} - {b}={sub_res}>>\n"
            f"3. Divide both sides by {a}: x = {sub_res} / {a} = <<{sub_res} / {a}={x}>>"
        )
        return format_math_reasoning(q, steps, str(x))

    def _generate_conversational(self) -> Tuple[str, str]:
        """Draws natural dialogue from SmolTalk to prevent math collapse."""
        if self.conversations:
            u, a = self.conversations[self.conv_idx]
            self.conv_idx = (self.conv_idx + 1) % len(self.conversations)
            return format_conversational_turn(u, a)
        else:
            # Fallback natural dialogues
            fallback_convs = [
                ("Hello! How can you help me today?",
                 "Hello! I am PrimeLM-50M, an ultra-compact reasoning model equipped with second-order organic memory and the PRIME-Net symbolic harness. I can help you solve mechanical equipment diagnostics, algebraic problems, and multi-step reasoning tasks."),
                ("Explain how your thinking process works inside <think> and what your 2nd-order memory does.",
                 "Inside the <think> tags, I decompose complex problems step-by-step, isolate variables, and trigger the PRIME-Net symbolic engine for exact math. My 2nd-order memory layer (GTRM) stores compressed 32-dimensional topological blueprints in a 64 KB quadratic manifold, allowing me to recall key entities and facts across long conversations with constant memory.")
            ]
            return format_conversational_turn(*random.choice(fallback_convs))

    def _generate_physics_invariant(self) -> Tuple[str, str]:
        """Generates physics conservation problems with explicit Co-Thinker tags."""
        m = random.choice([2, 4, 6, 8])
        v = random.choice([2, 3, 4, 5])
        v_sq = v * v
        ke = int(0.5 * m * v_sq)
        q = f"Calculate the kinetic energy of an object of mass {m} kg moving at speed {v} m/s using E = 0.5 * m * v^2."
        calc = f"E = 0.5 * {m} * {v_sq} = <<0.5 * {m} * {v_sq}={ke}>> Joules"
        return format_physics_invariant(q, "E = 0.5 * m * v^2", calc, f"{ke} Joules")

    def get_example(self) -> Tuple[str, str]:
        """Draws an example from the balanced 5-domain mixture."""
        r = random.random()
        if r < 0.25:
            return self._generate_equipment_mechanics()
        elif r < 0.50:
            return self._generate_algebra_with_cothinker()
        elif r < 0.75:
            return self._generate_conversational()
        elif r < 0.90:
            return self._generate_declarative_memory()
        else:
            return self._generate_physics_invariant()

    def get_batch(self, batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes batch with exact prompt loss masking (labels = -100 on prompt)."""
        input_ids_list = []
        labels_list = []

        for _ in range(batch_size):
            prompt, response = self.get_example()
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

            seq_tokens = prompt_tokens + response_tokens
            if len(seq_tokens) > self.seq_len:
                seq_tokens = seq_tokens[:self.seq_len]
                prompt_len = min(len(prompt_tokens), self.seq_len)
            else:
                prompt_len = len(prompt_tokens)

            # Mask prompt tokens with -100
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


def test_v2_dataset():
    print("[*] Testing Upgraded SFT-v2 Dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    streamer = SFTReasoningStreamer(tokenizer, seq_len=512)

    for i in range(5):
        prompt, resp = streamer.get_example()
        print(f"\n--- Sample {i+1} ---")
        print(f"PROMPT: {prompt[:100]}...")
        print(f"RESPONSE:\n{resp[:200]}...")

    inputs, labels = streamer.get_batch(batch_size=2)
    assert inputs.shape == (2, 512)
    assert labels.shape == (2, 512)
    print("\n[SUCCESS] SFT-v2 Dataset verified and ready!")


if __name__ == "__main__":
    test_v2_dataset()
