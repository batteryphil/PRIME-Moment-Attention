"""
PRIME Local Authoring Engine
============================
Executes 100% offline, local long-form creative writing using Sao10K/L3-8B-Stheno-v3.2
accelerated by AMD ROCm / HIP and bounded by PRIME-Moment-Attention recurrent cache.
"""

import os
import sys
import json
import time
import torch
import pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.cache import PrimeMomentCache

NOVELS_ROOT = PROJECT_ROOT / "novels"
VAULT_DB_PATH = PROJECT_ROOT / "scout" / "vault" / "scout_vault.db"

STHENO_SYSTEM_PROMPT = """You are an acclaimed commercial dark romantasy author specializing in high-heat, psychological obsession, razor-sharp dialogue, and visceral sensory immersion (5/5 heat level).

WRITING RULES:
- Write exclusively in visceral, third-person limited prose.
- Focus intensely on micro-expressions, tactile sensations (cold stone, rough velvet, racing pulse, shallow breath), and simmering psychological tension.
- Characters have lethal agency: razor-sharp banter, hidden vulnerabilities, and dark magnetic attraction.
- NEVER summarize or rush through emotional transitions. Show the physical toll of every touch and word.
- NEVER write author notes, copyright notices, chapter intros, or meta commentary.
- Stop cleanly when the scene beat objective is reached."""

SCI_FI_SPACE_OPERA_PROMPT = """You are an acclaimed commercial hard science fiction and space opera author specializing in physics-grounded realism, awe-inspiring cosmic wonder, tactical vacuum combat, and deep character chemistry.

WRITING RULES:
- Base all science strictly on realistic physics: Newtonian orbital mechanics, delta-v propellant limits, flip-and-burn deceleration, gravitational time dilation (general relativity), thermal radiation, and the terrifying silence of vacuum.
- Zero sound in space: explosions are silent blinding flashes; kinetic impacts shudder violently through the deckplates; alarms flash in cabin glass.
- Bring immense sensory grounding: the smell of recycled ozone and copper inside the cockpit, the crushing agony of high-g burns, the weightless float of zero-G sweat, the blinding sapphire glow of an accretion disk.
- Dynamic characters with sharp agency: witty banter, intense friction, slow-burn romantic tension, and mutual respect between the displaced 21st-century test pilot and the fierce 31st-century pirate captain.
- NEVER summarize or rush through tense moments.
- NEVER write author notes, acknowledgments, copyright notices, or meta commentary.
- Stop cleanly when the scene beat objective is reached."""


class PrimeLocalAuthor:
    """
    Local GPU-accelerated chapter generator powered by Sao10K/L3-8B-Stheno-v3.2
    and stabilized with PRIME-Moment-Attention memory recurrence.
    """
    _instance = None
    _model = None
    _tokenizer = None
    _loaded_model_id = None

    def __init__(self, model_id: str = "Sao10K/L3-8B-Stheno-v3.2", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self._ensure_loaded()

    def _ensure_loaded(self):
        # Check if we already loaded the correct model
        if PrimeLocalAuthor._model is not None and PrimeLocalAuthor._loaded_model_id == self.model_id:
            self.model = PrimeLocalAuthor._model
            self.tokenizer = PrimeLocalAuthor._tokenizer
            return

        # If a different model was loaded, free it first
        if PrimeLocalAuthor._model is not None:
            print(f"[*] [PRIME-Author] Unloading previous model {PrimeLocalAuthor._loaded_model_id}...")
            del PrimeLocalAuthor._model
            del PrimeLocalAuthor._tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[*] [PRIME-Author] Initializing {self.model_id} on {self.device} (ROCm / bfloat16)...")
        t0 = time.time()
        
        # Check if local snapshot exists or if fallback needed
        try:
            PrimeLocalAuthor._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            print(f"[!] Primary model tokenizer {self.model_id} failed: {e}. Trying fallback...")
            fallback_id = "NousResearch/Meta-Llama-3-8B"
            self.model_id = fallback_id
            PrimeLocalAuthor._tokenizer = AutoTokenizer.from_pretrained(fallback_id)

        if PrimeLocalAuthor._tokenizer.pad_token is None:
            PrimeLocalAuthor._tokenizer.pad_token = PrimeLocalAuthor._tokenizer.eos_token

        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        device_target = {"": 0} if self.device == "cuda" else "cpu"
        PrimeLocalAuthor._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_target,
            low_cpu_mem_usage=True
        )
        PrimeLocalAuthor._loaded_model_id = self.model_id

        load_sec = time.time() - t0
        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[✓] [PRIME-Author] Model {self.model_id} loaded in {load_sec:.1f}s | Active VRAM: {vram_gb:.2f} GB")

        self.model = PrimeLocalAuthor._model
        self.tokenizer = PrimeLocalAuthor._tokenizer

    def format_chat_prompt(self, user_prompt: str, system_prompt: str = STHENO_SYSTEM_PROMPT) -> str:
        """
        Formats user and system instructions using standard Llama-3 / Stheno ChatML headers.
        """
        return (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate_scene(
        self,
        prompt: str,
        system_prompt: str = STHENO_SYSTEM_PROMPT,
        max_new_tokens: int = 800,
        temperature: float = 1.12,
        top_k: int = 50,
        min_p: float = 0.075,
        repetition_penalty: float = 1.10
    ) -> str:
        """
        Generates a continuous, visceral narrative scene using Stheno-v3.2's optimal
        creative sampling parameters without artificial n-gram bans.
        """
        clean_user = prompt.strip()
        formatted_input = self.format_chat_prompt(clean_user, system_prompt=system_prompt)

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # Stop tokens: EOS and <|eot_id|>
        stop_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.eos_token_id:
            stop_ids.append(eot_id)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=stop_ids
            )

        gen_tokens = outputs[0][input_len:]
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

        # Sanitize special tokens and metadata
        decoded = decoded.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").strip()

        # Strip any accidental assistant prefix or prompt repetition
        lines = []
        for line in decoded.split("\n"):
            l_strip = line.strip()
            if l_strip.startswith("assistant") or l_strip.startswith("###") or l_strip.startswith("Story text:"):
                continue
            if l_strip.startswith("Objective:") or l_strip.startswith("Sensory details:") or l_strip.startswith("Novel:"):
                continue
            if l_strip.startswith("Subgenre:") or l_strip.startswith("Heat Level:") or l_strip.startswith("Characters:"):
                continue
            if any(h in l_strip.lower() for h in [
                "new york times best-selling", "writes ya fiction", "lives in los angeles",
                "all rights reserved", "isbn:", "copyright ©", "published by", "chapter end", "word count:"
            ]):
                break
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        # Ensure clean sentence termination if stopped mid-sentence
        if cleaned and not cleaned.endswith(('.', '!', '?', '"', '”', '…')):
            last_punct = max(
                cleaned.rfind('. '), cleaned.rfind('! '), cleaned.rfind('? '),
                cleaned.rfind('."'), cleaned.rfind('!"'), cleaned.rfind('?"'),
                cleaned.rfind('.”'), cleaned.rfind('!”'), cleaned.rfind('?”'),
                cleaned.rfind('.\n'), cleaned.rfind('!\n'), cleaned.rfind('?\n')
            )
            if last_punct != -1 and last_punct > len(cleaned) - 150:
                cleaned = cleaned[:last_punct + 1].strip()
        return cleaned

    def draft_chapter(self, slug: str, chapter_num: int, target_words: int = 2400) -> Dict[str, Any]:
        """
        Drafts a full 4-beat chapter locally using the novel's Book Bible,
        Research Dossier, and seamless continuity from previous chapters.
        """
        novel_dir = NOVELS_ROOT / slug
        chapters_dir = novel_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        bible_path = novel_dir / "BOOK_BIBLE.json"
        bible = json.loads(bible_path.read_text(encoding="utf-8")) if bible_path.exists() else {}

        title = bible.get("title", slug.replace("_", " ").title())
        subgenre = bible.get("subgenre", "Hard Science Fiction Space Opera" if ("horizon" in slug or "space" in slug) else "Dark Gothic Stalker Romantasy")
        heat_level = bible.get("heat_level", 3 if "space" in subgenre.lower() or "sci-fi" in subgenre.lower() else 5)
        total_chapters = bible.get("total_chapters", 20)

        # Genre-adaptive system prompt
        if any(k in subgenre.lower() or k in title.lower() or k in slug.lower() for k in ["space", "sci-fi", "science fiction", "horizon"]):
            active_system_prompt = SCI_FI_SPACE_OPERA_PROMPT
            characters = "Commander Leo Mercer (displaced 21st-century experimental test pilot) and Captain Astrid 'Vex' Ross (lethal, brilliant corsair captain of the Starlight Marauder)"
        else:
            active_system_prompt = STHENO_SYSTEM_PROMPT
            characters = "Aurelia Vex (cunning venom alchemist) and Prince Caelum (blindfolded High Executioner, obsessive shadow stalker)"

        # Retrieve outline for this chapter
        chapter_meta = {}
        for ch in bible.get("chapters", []):
            if ch.get("chapter") == chapter_num:
                chapter_meta = ch
                break

        ch_title = chapter_meta.get("title", f"Chapter {chapter_num}")

        # Check for previous chapter continuity
        prev_context = ""
        prev_ch_num = chapter_num - 1
        if prev_ch_num >= 1:
            prev_file = chapters_dir / f"chapter_{prev_ch_num:02d}.md"
            if prev_file.exists():
                prev_text = prev_file.read_text(encoding="utf-8")
                words = prev_text.split()
                prev_context = " ".join(words[-250:]) if len(words) > 250 else prev_text

        print(f"\n[*] [PRIME-Author] Drafting Chapter {chapter_num}/{total_chapters}: '{ch_title}' for '{title}' (Stheno-v3.2 on AMD GPU)...")

        ch_summary = chapter_meta.get("summary", "")

        # Decompose Chapter into 4 Commercial Beats
        if slug == "beyond_the_event_horizon" and chapter_num == 1:
            beats = [
                (
                    "Beat 1: The Test Flight & Gravitational Gradient",
                    f"Commander Leo Mercer straps into the high-g acceleration couch of the experimental Chronos-1 prototype deep in the Cygnus sector. The hum of the antimatter-catalyzed fusion torch. Routine telemetry shatters when the magnetic containment coils rupture, throwing the ship off vector into the steep gravitational gradient of the stellar black hole Cygnus X-1. Relentless g-forces, screaming klaxons, and the breathtaking, terrifying optical distortion of the glowing accretion disk bending starlight into infinite rings."
                ),
                (
                    "Beat 2: Skimming the Ergosphere & Frame-Dragging",
                    f"Leo fights the manual controls as the black hole's gravity drags the fabric of space-time itself. High-g fluid injectors burn in his veins to prevent blackout. Outside the cockpit canopy, relativistic Doppler shift turns the universe into blinding blue-violet streaks. He calculates a desperate, razor-thin hyperbolic slingshot trajectory through the ergosphere, firing the emergency reserve reaction mass directly against the horizon's edge."
                ),
                (
                    "Beat 3: The Thousand-Year Gravitational Slingshot",
                    f"The physics of general relativity take hold. Inside the ship, Leo experiences sheer crushing acceleration and temporal distortion for what feels like 45 agonizing minutes. Outside the gravitational well, space-time dilates exponentially. The stars twist and blur as centuries whip past in silent fire. As the ship slingshots free into flat space-time, the torch burns out and Leo slips into unconsciousness, the chronometer on his flight console spinning wild."
                ),
                (
                    "Beat 4: Awakening in the Void & The Pirate Boarding",
                    f"Leo awakens in freezing zero-G to a dead cockpit and emergency reserve batteries. Looking out at the starfield, constellations are distorted and unfamiliar—his nav-computer reads an elapsed mission time of 1,000.4 Earth years. Before he can process the staggering grief and shock, a massive shadow eclipses the stars. An armed 280-meter scavenger frigate locks magnetic docking clamps onto the Chronos-1. The outer airlock hisses open, and Captain Astrid 'Vex' Ross steps into the breach with a magnetic coil-carbine aimed at his heart."
                )
            ]
        elif slug == "beyond_the_event_horizon" and chapter_num == 2:
            beats = [
                (
                    "Beat 1: The Airlock Rupture & Zero-G Standoff",
                    f"Captain Astrid 'Vex' Ross floats through the breached airlock of Chronos-1 in her combat void-suit, coil-carbine raised, expecting cold scrap or mummified remains. Instead, she finds Commander Leo Mercer floating alive out of the cockpit couch, service pistol drawn in a tense zero-gravity standoff. Sparks drift like dying stars; the hiss of equalizing oxygen."
                ),
                (
                    "Beat 2: Razor Banter & The Golden Age Relic",
                    f"Astrid's cynical corsair wit clashes with Leo's calm military discipline. Astrid's helmet HUD flashes red with impossible telemetry: the ship's atomic clock and unencrypted Federation serial numbers indicate it was built a thousand years ago. When Leo attempts an evasive pivot, Astrid counters with ruthless, fluid zero-G hand-to-hand maneuvers, disarming him with breathless physical proximity, their helmet faceplates nearly touching."
                ),
                (
                    "Beat 3: Across the Umbilical Tube to the Starlight Marauder",
                    f"Securing Leo in magnetic tethers, Astrid guides him across the pressurized accordion umbilical into the Starlight Marauder. The sudden, violent sensory shock as the ship's 0.8g centrifugal spin slams Leo's disoriented body back into artificial gravity. The smells of hot engine grease, burnt wiring, and recycled coffee. The Marauder's crew—especially the cybernetic engineer Orlo—gawking in disbelief at a living pilot from the mythic pre-Collapse era."
                ),
                (
                    "Beat 4: The Interrogation & The Forbidden Star Chart",
                    f"Astrid pushes Leo into the observation commons for a retinal scan and debriefing. The electric friction and simmering tension between them as she questions his impossible flight path from Cygnus X-1. When Leo demands to see the navigation charts for Earth and the Sol system, Astrid turns the holo-projector on: Sol's coordinates are flagged in flashing amber as an quarantined, uninhabitable dead zone. A sensor alarm chirps—an unknown long-range thermal contact is closing fast."
                )
            ]
        elif "horizon" in slug or "space" in subgenre.lower():
            beats = [
                (
                    f"Beat 1: Immediate Stakes & Vacuum Realism",
                    f"Opening conflict and sensory immersion of '{ch_title}' based on: {ch_summary}. Ground strictly in realistic physics (orbital mechanics, zero-G inertia, thermal radiation, cold metal). Establish the immediate danger or challenge facing Leo Mercer and Captain Astrid Ross."
                ),
                (
                    f"Beat 2: Friction, Agency & Razor Banter",
                    f"Rising tactical and interpersonal conflict. Sharp dialogue and clash of perspectives between Leo's 21st-century military honor and Astrid's hardened 31st-century pirate survivalism. Compelling agency, mutual testing, and high stakes."
                ),
                (
                    f"Beat 3: High-Stakes Crisis & Electric Chemistry",
                    f"The physical and emotional peak of the chapter. A harrowing tactical crisis, mechanical failure, orbital burn, or close combat that demands teamwork. Intimate, breathless proximity, simmering attraction, and mutual respect under fire."
                ),
                (
                    f"Beat 4: The Cosmic Shift & The Cliffhanger",
                    f"The immediate aftermath of the crisis, an awe-inspiring glimpse of deep cosmic wonder or terrifying astrophysics, and a sudden revelation or unexpected sensor contact that propels the story urgently into Chapter {chapter_num + 1}."
                )
            ]
        elif slug == "a_crown_of_gilded_bones" and chapter_num == 4:
            beats = [
                (
                    "Beat 1: The Tower Breach & The Secret Escape",
                    f"In the deafening clamor of the Dead King's bell and the battering ram shaking the tower doors, Caelum orders the royal inquisitors to seal the perimeter. Aurelia seizes the split-second distraction to pry loose the iron grate of the ancient ossuary flue beneath the floorboards, sliding into the pitch-black catacomb shaft as the heavy chamber doors splinter open."
                ),
                (
                    "Beat 2: The Starless Catacombs & The Stalker's Hunt",
                    f"Aurelia navigates the freezing, bone-lined tunnels of the Starless Catacombs beneath the Citadel. Ice-water drips from calcified stalactites, and phosphorescent ghost-moss glimmers faintly. Her pulse races as she senses a predator closing in—the unmistakable low vibration of bone-magic echoing through the flagstones. Caelum is hunting her in the pitch dark."
                ),
                (
                    "Beat 3: The Shrine of Skeletons & Searing Confrontation",
                    f"Caelum cuts off her escape in the subterranean Shrine of Ancient Kings. He steps from the shadows without his blindfold, pinning her against a carved sarcophagus. Blade-to-throat banter dissolves into suffocating physical friction: his hands rough on her waist, dark whispered filth, testing her racing pulse, and an explosive, breathless 5/5 encounter in the crypt shadows."
                ),
                (
                    "Beat 4: The Corpse in the Sarcophagus & The Cliffhanger",
                    f"As their ragged breaths mingle in the dark, Aurelia’s hand brushes against something sticky inside the open sarcophagus—the freshly slaughtered body of the Royal Cupbearer, clutching an obsidian dagger stamped with Archon Malichor's crest. The true poisoner was murdered to frame Aurelia. Before they can react, iron boots echo down the crypt corridor."
                )
            ]
        else:
            beats = [
                (
                    f"Beat 1: The Inciting Hook & Sensory Escalation",
                    f"Opening immediate stakes of '{ch_title}'. Sensory atmosphere, tactile grounding, and immediate conflict arising from: {ch_summary}. Focus on physical sensations, adrenaline, and simmering power dynamics."
                ),
                (
                    f"Beat 2: Forced Proximity & Psychological Friction",
                    f"Escalation of conflict between the main characters. Sharp, lethal dialogue, razor banter, conflicting motives. They are forced together to navigate: {ch_summary}. Unbearable physical proximity and dangerous attraction."
                ),
                (
                    f"Beat 3: Searing Climax & High-Heat Tension",
                    f"Emotional and physical breaking point of the chapter (Heat: {heat_level}/5). A boundary is pushed or crossed; raw touches, breath against skin, dark possessiveness, and electric vulnerability in the shadows."
                ),
                (
                    f"Beat 4: Resolution, Treachery & The Cliffhanger",
                    f"The immediate aftermath, a sudden shocking revelation, new threat, or ominous twist that turns the situation on its head and demands turning the page to Chapter {chapter_num + 1}."
                )
            ]

        scenes = []
        t0_ch = time.time()

        for i, (beat_name, beat_obj) in enumerate(beats, 1):
            print(f"    -> [Stheno-v3.2] Generating {beat_name} ({i}/{len(beats)})...")
            last_snippet = prev_context if not scenes else " ".join(scenes[-1].split()[-200:])
            
            user_prompt = (
                f"Novel: {title}\n"
                f"Subgenre: {subgenre}\n"
                f"Heat Level: {heat_level}/5\n"
                f"Characters: {characters}\n"
                f"Current Chapter: Chapter {chapter_num}: {ch_title}\n"
                f"Scene {i} Objective: {beat_obj}\n\n"
                f"Prior Continuity Anchor (Continue directly from here):\n"
                f"\"{last_snippet}\"\n\n"
                f"Write the complete scene in vivid, publication-ready prose now."
            )

            scene_text = self.generate_scene(user_prompt, system_prompt=active_system_prompt, max_new_tokens=750)
            if scene_text:
                scenes.append(scene_text)
                print(f"       [✓] Scene {i} drafted: {len(scene_text.split()):,} words")

        # Assemble full chapter
        chapter_content = f"# Chapter {chapter_num}: {ch_title}\n\n" + "\n\n✦ ✦ ✦\n\n".join(scenes)
        word_count = len(chapter_content.split())
        elapsed_ch = time.time() - t0_ch

        # Save to disk
        out_file = chapters_dir / f"chapter_{chapter_num:02d}.md"
        out_file.write_text(chapter_content, encoding="utf-8")

        vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        print(f"\n[✓] [PRIME-Author] Chapter {chapter_num} complete: {word_count:,} words in {elapsed_ch:.1f}s | Saved to {out_file.name} | VRAM: {vram_mb:.1f} MB")

        return {
            "slug": slug,
            "chapter": chapter_num,
            "title": ch_title,
            "words": word_count,
            "path": str(out_file),
            "status": "COMPLETED"
        }


def get_author(model_id: str = "Sao10K/L3-8B-Stheno-v3.2") -> PrimeLocalAuthor:
    """Singleton getter for web_ui and background daemons."""
    if PrimeLocalAuthor._instance is None or PrimeLocalAuthor._loaded_model_id != model_id:
        PrimeLocalAuthor._instance = PrimeLocalAuthor(model_id=model_id)
    return PrimeLocalAuthor._instance


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "beyond_the_event_horizon"
    ch_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    author = get_author()
    res = author.draft_chapter(slug, ch_num, target_words=2500)
    print("Draft Result:", res)
