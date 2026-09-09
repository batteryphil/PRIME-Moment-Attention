"""
Romantasy Dataset Builder for Local LoRA Training
=================================================
Compiles high-heat dark romance / romantasy training samples formatted for
instruction-tuned Llama-3 / Stheno models.
"""

import json
import pathlib
from typing import List, Dict, Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "romantasy_corpus"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def build_curated_samples() -> List[Dict[str, Any]]:
    """
    Returns curated dark romantasy training pairs focusing on:
    - High-heat romantic tension (5/5)
    - Stalker / obsessive love tropes
    - Visceral sensory details & sharp dialogue
    - Zero meta-commentary / clean fiction
    """
    samples = [
        {
            "instruction": "Novel: A Crown of Gilded Bones\nCharacters: Aurelia Vex and Prince Caelum\nScene Objective: Forced proximity inside the carriage as Caelum pins Aurelia against the velvet cushions, testing her pulse and scenting her fear.",
            "completion": (
                "The carriage lurched over a frost-heaved rut, throwing Aurelia forward into a wall of living shadow. "
                "Caelum didn’t flinch. His hand shot out, broad fingers locking around her throat—not with the cruelty of an executioner, "
                "but with the deliberate, suffocating weight of a predator claiming what had wandered into his den.\n\n"
                "\"Careful, little serpent,\" his voice rumbled against the shell of her ear, deep and vibrating with the dark resonance of osteomancy. "
                "His breath smelled of bitter clove and mountain snow. \"Break that slender neck on the floorboards, and you rob me of the pleasure of wringing the truth from you myself.\"\n\n"
                "Her pulse thrashed violently against his thumb. He leaned closer until his blindfold grazed her cheek, the silver silk rough against her fevered skin. "
                "She should have reached for the hemlock needle tucked beneath her cuff. She should have driven it through his ribs. "
                "Instead, the heat pouring off his broad chest pinned the air in her lungs, thick and intoxicating.\n\n"
                "\"You won't kill me,\" Aurelia whispered, her chin lifting even as his grip tightened, pressing her back into the crushed black velvet. "
                "\"The court is starving for your bone-magic, Prince. Without my venom to temper the calcification, your marrow will turn to glass before the solstice.\"\n\n"
                "A slow, lethal smile curved Caelum’s lips. His thumb smoothed over the fluttering hollow beneath her jaw, savoring the frantic gallop of her blood. "
                "\"Who said anything about killing you, Aurelia? Death is an escape. And I intend to keep you where I can feel every breath you draw.\""
            )
        },
        {
            "instruction": "Novel: The Shadowed Vow\nCharacters: Isla and Rowan\nScene Objective: A moonlit balcony duel where blade-to-throat standoff transitions into irresistible, breathless physical intimacy.",
            "completion": (
                "Rowan’s daggers were at her throat before she could draw breath to scream. "
                "The steel bit cold into the hollow of her collarbone, but the heat radiating from his broad shoulders burned through the thin silk of her night rail.\n\n"
                "\"You shouldn't have come back, Isla,\" he murmured, his voice a low scrape of gravel against flint. "
                "He stepped into her guard, eliminating the last sliver of air between them. Every line of his body was coiled muscle and lethal intent, "
                "yet his eyes tracked the frantic rise and fall of her chest with an intensity that bordered on madness.\n\n"
                "\"Then strike,\" she challenged, daring to step into the blade until a single red bead welled beneath the edge. "
                "\"Cut me down, Rowan. Or admit that you followed my scent across three kingdoms because you couldn't bear to sleep without the sound of my heartbeat.\"\n\n"
                "His pupils dilated until the dark amber of his irises vanished entirely. With a guttural curse, he tossed the dagger over the marble balustrade. "
                "The metal clattered far below in the courtyard, forgotten before it even hit the stones. "
                "His hands seized her hips, hauling her up against him so hard the breath rushed from her parted lips.\n\n"
                "\"You think this is mercy?\" he growled against her mouth, his lips brushing hers with agonizing restraint. "
                "\"This is possession. And by the time the dawn breaks, you won't even remember your own name.\""
            )
        },
        {
            "instruction": "Novel: Court of Crimson Feathers\nCharacters: Vesper and Lord Malakai\nScene Objective: The blood oath ritual where Malakai binds Vesper's soul to his in the subterranean shrine.",
            "completion": (
                "The subterranean temple was colder than the grave, but Vesper’s skin burned as if bathed in liquid flame. "
                "Malakai knelt before her on the black basalt altar, his long, ink-stained fingers tracing the vein that ran from her inner wrist to her elbow. "
                "Behind his high cheekbones, the faint violet glow of his void-magic pulsed in time with her own racing heart.\n\n"
                "\"Give me your hand, little bird,\" he commanded softly. It wasn't a request. It was an inevitable gravity that she had ceased trying to fight.\n\n"
                "She placed her palm in his. His skin was fever-hot, contrasting violently with the damp chill of the subterranean vault. "
                "With the tip of an obsidian quill, he scored a delicate line across her palm, then did the same to his own. "
                "When their wounded hands pressed together, the snap of bound souls sounded like thunder inside her skull.\n\n"
                "Vesper gasped, arching off the stone as his essence flooded her veins—dark, ancient, and completely possessive. "
                "Malakai’s free arm wrapped around her waist, pulling her flush against his chest to steady her trembling frame. "
                "His forehead rested against hers, their mingled breaths coming in ragged, synchronization.\n\n"
                "\"Now you are mine,\" he whispered against her trembling lips, his voice trembling with a raw reverence that terrified her more than his blade ever could. "
                "\"In every life, in every shadow, until the stars themselves burn to ash.\""
            )
        }
    ]
    return samples


def export_dataset_json(out_path: str = None) -> str:
    if out_path is None:
        out_path = str(DATASET_DIR / "romantasy_train.json")

    raw_samples = build_curated_samples()
    formatted = []

    for s in raw_samples:
        text = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a master commercial dark romantasy author specializing in high-heat, psychological obsession, razor-sharp dialogue, and multi-sensory immersion (5/5 heat level).\n"
            f"Write exclusively in visceral, third-person limited prose. Never summarize. Never write author notes.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{s['instruction']}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{s['completion']}<|eot_id|>"
        )
        formatted.append({"text": text, "instruction": s["instruction"], "completion": s["completion"]})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2)

    print(f"[✓] [Dataset] Exported {len(formatted)} training samples to {out_path}")
    return out_path


if __name__ == "__main__":
    export_dataset_json()
