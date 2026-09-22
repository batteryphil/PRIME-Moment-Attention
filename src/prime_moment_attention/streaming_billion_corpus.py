#!/usr/bin/env python3
"""
Streaming Billion-Token Multi-Corpus Engine
===========================================
High-throughput, zero-copy streaming pipeline for PrimeLM-50M.
Combines 5 diverse local data streams into contiguous L=1024 sequences:
  1. SmolTalk Ultra (30%): Reasoning, dialogues, algorithms, multi-turn QA
  2. TinyStories (25%): Grammatical diversity, syntactic fluency, common sense
  3. Physical & Algebraic Invariants (20%): Thermodynamics, kinetic energy, circuits, linear systems
  4. Mathematics & Logic (15%): GSM8K multi-step word problems & derivations
  5. Science QA (10%): AI2-ARC challenge & easy science questions
"""

import os
import glob
import random
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

class MultiCorpusBillionStreamer:
    def __init__(self, seq_len=1024, buffer_target=100000, seed=42):
        self.seq_len = seq_len
        self.buffer_target = buffer_target
        random.seed(seed)
        
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token
        self.eos_id = self.tok.eos_token_id
        
        # 1. Discover Shards
        self.smol_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--HuggingFaceTB--smoltalk/**/train-*.parquet"), recursive=True))
        self.tiny_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/**/train-*.parquet"), recursive=True))
        self.gsm_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--openai--gsm8k/**/train-*.parquet"), recursive=True))
        self.arc_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--ai2_arc/**/train-*.parquet"), recursive=True))
        
        # Shard pointers
        self.smol_idx = 0
        self.smol_table = None
        self.smol_row = 0
        
        self.tiny_idx = 0
        self.tiny_table = None
        self.tiny_row = 0
        
        # In-memory datasets (fast random access)
        self.gsm_items = []
        if self.gsm_files:
            t = pq.read_table(self.gsm_files[0])
            for i in range(len(t)):
                self.gsm_items.append(f"Problem: {t['question'][i].as_py()}\nSolution:\n{t['answer'][i].as_py()}")
                
        self.arc_items = []
        for af in self.arc_files:
            t = pq.read_table(af)
            for i in range(len(t)):
                self.arc_items.append(f"Science Question: {t['question'][i].as_py()}\nAnswer: {t['answerKey'][i].as_py()}")
                
        # Token & Invariant Buffers
        self.token_buffer = []
        self.inv_buffer = []
        
        self._load_next_smol_shard()
        self._load_next_tiny_shard()
        
    def _load_next_smol_shard(self):
        if not self.smol_files: return
        f = self.smol_files[self.smol_idx % len(self.smol_files)]
        self.smol_table = pq.read_table(f, columns=["messages"])
        self.smol_row = 0
        self.smol_idx += 1

    def _load_next_tiny_shard(self):
        if not self.tiny_files: return
        f = self.tiny_files[self.tiny_idx % len(self.tiny_files)]
        self.tiny_table = pq.read_table(f, columns=["text"])
        self.tiny_row = 0
        self.tiny_idx += 1

    def _sample_procedural_invariant(self):
        mode = random.randint(0, 4)
        if mode == 0:
            # Linear Equations
            a = random.randint(2, 12)
            x = random.randint(2, 50)
            b = random.randint(1, 50)
            c = a * x + b
            txt = f"Problem: Solve for x in {a} * x + {b} = {c}.\nStep 1: Subtract {b} from both sides to get {a} * x = {c - b}.\nStep 2: Divide both sides by {a} to get x = {x}.\nFinal Answer: {x}."
            inv = [float(a), float(b), float(c), float(x)]
        elif mode == 1:
            # Thermodynamics Q = m * c * deltaT
            m = random.randint(1, 15)
            dT = random.randint(5, 60)
            c_spec = 4
            Q = m * c_spec * dT
            txt = f"Problem: Calculate heat required for {m} kg water heated by {dT} K.\nFormula: Q = m * c * deltaT.\nCalculation: Q = {m} * {c_spec} * {dT} = {Q} Joules.\nFinal Answer: {Q}."
            inv = [float(m), float(c_spec), float(dT), float(Q)]
        elif mode == 2:
            # Kinetic Energy E = 0.5 * m * v^2
            m = random.randint(2, 24)
            v = random.randint(2, 16)
            E = int(0.5 * m * (v ** 2))
            txt = f"Problem: Calculate kinetic energy of mass {m} kg moving at speed {v} m/s.\nFormula: E = 0.5 * m * v^2.\nCalculation: E = 0.5 * {m} * {v**2} = {E} Joules.\nFinal Answer: {E}."
            inv = [float(m), float(v), 0.5, float(E)]
        elif mode == 3:
            # Ohm's Law V = I * R
            I = random.randint(1, 20)
            R = random.randint(2, 40)
            V = I * R
            txt = f"Problem: Find electrical voltage across a circuit with current {I} A and resistance {R} Ohms.\nFormula: V = I * R.\nCalculation: V = {I} * {R} = {V} Volts.\nFinal Answer: {V}."
            inv = [float(I), float(R), 1.0, float(V)]
        else:
            # Proportional scaling y = k * x
            k = random.randint(2, 10)
            x = random.randint(3, 25)
            y = k * x
            txt = f"Problem: If 1 unit yields {k} items, how many items do {x} units yield?\nFormula: y = k * x.\nCalculation: y = {k} * {x} = {y} items.\nFinal Answer: {y}."
            inv = [float(k), float(x), 1.0, float(y)]
        return txt, inv

    def refill_buffer(self):
        batch_texts = []
        batch_invs = []
        
        while len(batch_texts) < 3000:
            p = random.random()
            if p < 0.30:
                # SmolTalk
                if self.smol_row >= len(self.smol_table):
                    self._load_next_smol_shard()
                msgs = self.smol_table["messages"][self.smol_row].as_py()
                self.smol_row += 1
                txt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in msgs])
                batch_texts.append(txt)
                batch_invs.append([0.0, 0.0, 0.0, 0.0])
            elif p < 0.55:
                # TinyStories
                if self.tiny_row >= len(self.tiny_table):
                    self._load_next_tiny_shard()
                story = self.tiny_table["text"][self.tiny_row].as_py()
                self.tiny_row += 1
                batch_texts.append(story)
                batch_invs.append([0.0, 0.0, 0.0, 0.0])
            elif p < 0.75:
                # Physical Invariants
                txt, inv = self._sample_procedural_invariant()
                batch_texts.append(txt)
                batch_invs.append(inv)
            elif p < 0.90:
                # GSM8K
                if self.gsm_items:
                    item = random.choice(self.gsm_items)
                    batch_texts.append(item)
                    batch_invs.append([0.0, 0.0, 0.0, 0.0])
            else:
                # ARC Science
                if self.arc_items:
                    item = random.choice(self.arc_items)
                    batch_texts.append(item)
                    batch_invs.append([0.0, 0.0, 0.0, 0.0])
                    
        # Tokenize batch
        for txt, inv in zip(batch_texts, batch_invs):
            tokens = self.tok.encode(txt) + [self.eos_id]
            self.token_buffer.extend(tokens)
            # Match token length with invariant conditioning
            for _ in range(len(tokens)):
                self.inv_buffer.append(inv)

    def get_batch(self, batch_size=8):
        req_tokens = batch_size * self.seq_len
        while len(self.token_buffer) < req_tokens + 1024:
            self.refill_buffer()
            
        # Extract contiguous tokens
        tok_chunk = self.token_buffer[:req_tokens]
        inv_chunk = self.inv_buffer[:req_tokens]
        self.token_buffer = self.token_buffer[req_tokens:]
        self.inv_buffer = self.inv_buffer[req_tokens:]
        
        # Reshape into [B, L]
        batch_ids = torch.tensor(tok_chunk, dtype=torch.long).view(batch_size, self.seq_len)
        
        # Extract invariant vector per sequence (take representative vector from middle token)
        seq_invs = []
        for b in range(batch_size):
            start = b * self.seq_len
            mid = start + self.seq_len // 2
            seq_invs.append(inv_chunk[mid])
            
        batch_invs = torch.tensor(seq_invs, dtype=torch.bfloat16)
        return batch_ids, batch_invs
