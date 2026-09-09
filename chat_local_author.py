#!/usr/bin/env python3
"""
PRIME Interactive Local Author Studio CLI
=========================================
Direct interactive environment on your desktop.
Talk directly to your local GPU-accelerated model (NousResearch/Meta-Llama-3-8B)
with PRIME-Moment-Attention memory recurrence.
"""

import os
import sys
import time
import torch
import readline
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scout.prime_local_author import PrimeLocalAuthor
from scout.romantasy_studio import RomantasyStudio, NOVELS_ROOT

def main():
    print("=" * 80)
    print("👑 PRIME Interactive Local Author Studio (100% Offline GPU)")
    print("🔥 Direct Control Room: Prompt your stories, scenes, and dark smut directly")
    print("💻 Powered by NousResearch/Meta-Llama-3-8B on AMD Radeon Graphics (ROCm)")
    print("=" * 80)
    print("\n[*] Loading local model onto AMD GPU (this takes ~10-15s on startup)...")
    
    author = PrimeLocalAuthor()
    
    print("\n[✓] Local author online and ready!")
    print("\nCommands:")
    print("  'exit' or 'quit'   - Exit the interactive studio")
    print("  'status'           - View active novel catalog and word counts")
    print("  'heat <1-5>'       - Set default spice/heat level (Current: 5/5 🌶️)")
    print("  'clear'            - Clear the terminal screen")
    print("\nOr simply type what you want the AI to write (e.g. 'Write a dark scene where Caelum catches Aurelia in the bathhouse with a knife...'):\n")
    
    heat_level = 5
    
    while True:
        try:
            prompt = input(f"\n[Phil @ Local-AI | Heat: {'🌶️'*heat_level}] > ").strip()
            if not prompt:
                continue
                
            if prompt.lower() in ["exit", "quit", "q"]:
                print("\n[!] Exiting Local Studio. Good writing!")
                break
                
            if prompt.lower() == "clear":
                os.system("clear")
                continue
                
            if prompt.lower() == "status":
                studio = RomantasyStudio()
                catalog = studio.get_library_catalog()
                print("\n--- Current Novel Catalog ---")
                for b in catalog:
                    pct = round((b['current_words'] / b['target_words']) * 100, 1) if b['target_words'] > 0 else 0
                    print(f"  • [{b['status']}] {b['title']} — {b['current_words']:,} / {b['target_words']:,}w ({pct}%) | Heat: {'🌶️'*b['heat_level']}")
                continue
                
            if prompt.lower().startswith("heat"):
                parts = prompt.split()
                if len(parts) > 1 and parts[1].isdigit():
                    heat_level = max(1, min(5, int(parts[1])))
                    print(f"[✓] Heat level set to {heat_level}/5 {'🌶️'*heat_level}")
                else:
                    print(f"Current heat level: {heat_level}/5")
                continue

            # Generate on local GPU
            print(f"\n[*] Generating directly on local GPU (bfloat16, ROCm, Heat {heat_level}/5)...")
            t0 = time.time()
            
            enhanced_prompt = (
                f"Instruction: {prompt}\n"
                f"Target Heat Level: {heat_level}/5 (High-heat dark romance / explicit tension / visceral sensory prose).\n"
                f"Tone: Intense, dangerous, obsessive, dark, and seductive."
            )
            
            response = author.generate_scene(enhanced_prompt, max_new_tokens=850, temperature=0.8, top_p=0.92)
            gen_sec = time.time() - t0
            words = len(response.split())
            wps = words / gen_sec if gen_sec > 0 else 0
            
            print("\n" + "─" * 80)
            print(f"📖 GENERATED SCENE ({words} words • {gen_sec:.1f}s • {wps:.1f} w/s • VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB):")
            print("─" * 80 + "\n")
            print(response)
            print("\n" + "─" * 80)
            
            # Offer save
            save = input("\nSave this scene to a file? [y/N]: ").strip().lower()
            if save == 'y':
                save_name = input("Enter file name (default: scene_output.md): ").strip() or "scene_output.md"
                out_path = PROJECT_ROOT / "novels" / save_name
                out_path.write_text(response, encoding="utf-8")
                print(f"[✓] Saved to {out_path}")
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n[!] Session interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[!] Generation error: {e}")

if __name__ == "__main__":
    main()
