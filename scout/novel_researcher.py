"""
PRIME Romantasy Research & Architecture Engine
Conducts deep mythological, trope, worldbuilding, and psychological research
to generate structured Book Bibles for novels ranging from 50k to 250k words.
"""

import json
import sqlite3
import pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
NOVELS_ROOT = PROJECT_ROOT / "novels"
VAULT_DB_PATH = PROJECT_ROOT / "scout" / "vault" / "scout_vault.db"

class RomantasyResearcher:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(VAULT_DB_PATH)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS novel_research_dossiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT,
            topic TEXT,
            research_notes TEXT,
            created_at TEXT
        );
        """)
        conn.commit()
        conn.close()

    def store_research(self, slug: str, topic: str, notes: str):
        conn = sqlite3.connect(VAULT_DB_PATH)
        cur = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()
        cur.execute("""
        INSERT INTO novel_research_dossiers (slug, topic, research_notes, created_at)
        VALUES (?, ?, ?, ?);
        """, (slug, topic, notes, now_str))
        conn.commit()
        conn.close()

    def generate_research_dossier(self, slug: str) -> Dict[str, Any]:
        """Generates comprehensive worldbuilding, mythology, and character research for a novel."""
        if slug == "a_crown_of_gilded_bones":
            dossier = {
                "slug": slug,
                "title": "A Crown of Gilded Bones",
                "target_words": 75000,
                "total_chapters": 30,
                "heat_level": 5,
                "subgenre": "Dark Gothic Stalker Romantasy (Elevated Cat & Mouse)",
                "research_topics": {
                    "mythology_and_lore": (
                        "Subterranean Fae court built within the fossilized ribcage of an ancient primordial wyrm (The Ossuary Citadel). "
                        "The Bone Fae draw power from osteomancy—resonating with the calcium and skeletal memories of ancient beasts. "
                        "A strict blood-caste system exists where High Royals have porcelain-white bones that resist all steel, while Common Fae have brittle bone-structures."
                    ),
                    "magic_system_rules": (
                        "1. Bone-Resonance: Prince Caelum is blind, but perceives the physical world through the vibration of living marrow, pulse-rates, and calcium. He can hear her heart skip across a banquet hall. "
                        "2. Poison-Weaving: Aurelia crafts botanical nerve toxins using subterranean fungi (Pale Maiden, Ghost-Bell). "
                        "3. The Blood-Debt Betrothal: An ancient alchemical treaty requiring the vassal clan to offer a bride to the royal executioner. If the bride dies within a year, the vassal province is forfeited."
                    ),
                    "character_psychology": (
                        "Aurelia Vex (Protagonist / The Little Viper): Cynical alchemist and poisoner. Fiercely autonomous, razor-witted, and lethal. She refuses to be a trembling prey; she sets venomous traps and wields a hidden bone dagger. "
                        "Prince Caelum (Male Lead / The Shadow Executioner): Ruthless, masked executioner who stalks corrupt nobles in the dark. Becomes dangerously, monomaniacally obsessed with Aurelia. He shadows her every step, leaving sinister carved tokens and whispering possessive promises in the dark."
                    ),
                    "trope_pacing": (
                        "Elevated 'Haunting Adeline' Cat-and-Mouse Dynamic. Ch 1-3: Unseen stalker in the catacombs; carving tokens left on her pillow; knife-to-throat first confrontation. "
                        "Ch 5-8: 'Run, little viper' primal chase in the bone gardens. Forced proximity in the executioner's bedchambers. "
                        "Ch 14: The iconic 'Who did this to you?' moment where Caelum violently executes a rival lord who bruised her wrist. "
                        "Ch 18-25: Escalating, unapologetic 5/5 dark spice—knife play, shadow-binding, breathless primal dominance and mutual psychological surrender before the coup."
                    )
                }
            }
        elif slug == "the_serpent_and_the_star_thief":
            dossier = {
                "slug": slug,
                "title": "The Serpent and the Star-Thief",
                "target_words": 120000,
                "total_chapters": 45,
                "heat_level": 5,
                "research_topics": {
                    "mythology_and_lore": (
                        "The Black Spire Astrologicum: A floating gothic university orbiting an eclipsed celestial void. "
                        "Students harness Starlight Runes drawn from dying constellations. The Serpent House hoards forbidden astral cartography to summon ancient stellar serpents."
                    ),
                    "magic_system_rules": (
                        "1. Astral Ingestion: Drinking distilled liquid starlight grants superhuman reflexes and spatial phasing, but risks burning the nervous system. "
                        "2. Constellation Binding: Two souls who synchronize their astral charts can share thoughts across light-years, but feel each other's physical wounds."
                    ),
                    "character_psychology": (
                        "Lyra Crow (Protagonist): Street thief who forged entry into the prestigious academy by stealing the Star of Ophiuchus. Despises the aristocratic elite. "
                        "Nicholas Ashwood (Male Lead): Cold, arrogant prodigy of House Ashwood. Haunted by the disappearance of his father in the void. Captivated by Lyra's raw, unrefined power."
                    ),
                    "trope_pacing": (
                        "Academic rivals to lovers. Forbidden library encounters after dark. Forced lab partnership. Sizzling tension over star-charts. "
                        "Dark academia mystery leading to an eldritch conspiracy and high-spice forbidden encounters."
                    )
                }
            }
        elif slug == "beneath_the_ashen_sun":
            dossier = {
                "slug": slug,
                "title": "Beneath the Ashen Sun",
                "target_words": 180000,
                "total_chapters": 65,
                "heat_level": 4,
                "research_topics": {
                    "mythology_and_lore": (
                        "The volcanic ash-wastes of Kael-Drakor. Ruled by dragon-riding warlords who bond with massive basalt drakes. "
                        "The Sun was extinguished two centuries ago by the Shadow Veil, leaving only geothermal vents and glowing magma lakes to sustain life."
                    ),
                    "magic_system_rules": (
                        "1. Drake-Bond: Neural link between rider and drake. Death of one blinds the other. "
                        "2. Sun-Singing: High priestesses channel dormant solar heat to purify volcanic ash into arable soil."
                    ),
                    "character_psychology": (
                        "Seraphina (Protagonist): Last Sun-Singer of the Western Canyons. Defiant, unyielding, refuses to bow to dragon-lords. "
                        "General Rhaegar (Male Lead): Supreme commander of the Drake Legion. Brutal, solitary, burdened with preventing the volcanic collapse of the continent."
                    ),
                    "trope_pacing": (
                        "Warlord and captive priestess. Mating bond discovery. War council clashes. Epic aerial battles. Breathless forced proximity in desert tents."
                    )
                }
            }
        else: # kingdom_of_rust_and_ruin (250k words)
            dossier = {
                "slug": slug,
                "title": "Kingdom of Rust and Ruin",
                "target_words": 250000,
                "total_chapters": 90,
                "heat_level": 5,
                "research_topics": {
                    "mythology_and_lore": (
                        "Continental epic across five warring kingdoms: The Iron Marches, The Sunken Glades, The Cobalt Spire, The Ash Wastes, and The Frost Veil. "
                        "Ancient mechanical colossi (Titans of Rust) buried beneath the earth begin awakening as alchemical decay consumes the soil."
                    ),
                    "magic_system_rules": (
                        "1. Alchemical Transmutation: Manipulation of elemental metals and blood-alloys. "
                        "2. Necro-Mechanics: Reanimating titan machinery using soul-threads. "
                        "3. The Triple Covenant: Ancient royal bloodlines whose mating bonds dictate the magnetic stability of the continent."
                    ),
                    "character_psychology": (
                        "Multi-POV epic with three central couples representing different Romantasy dynamics: "
                        "Couple 1: Warrior Empress x Captive Rebel Warlord (Power dynamic, high spice). "
                        "Couple 2: Rogue Alchemist x Lethal Inquisitor (Enemies-to-lovers mystery). "
                        "Couple 3: Exiled Prince x Wildwoods Shapeshifter (Fated mates, primal heat)."
                    ),
                    "trope_pacing": (
                        "Massive 90-chapter, 4-Act structure. Grand military campaigns, political treaties, assassinations, court balls, and deeply emotional romantic payoffs."
                    )
                }
            }

        # Store research into database
        book_dir = NOVELS_ROOT / slug
        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "RESEARCH_DOSSIER.json").write_text(json.dumps(dossier, indent=2), encoding="utf-8")

        for topic, notes in dossier["research_topics"].items():
            self.store_research(slug, topic, notes)

        print(f"[+] Researched & Generated Dossier for: '{dossier['title']}' ({dossier['target_words']:,} words, {dossier['total_chapters']} chapters)")
        return dossier

if __name__ == "__main__":
    researcher = RomantasyResearcher()
    for slug in ["a_crown_of_gilded_bones", "the_serpent_and_the_star_thief", "beneath_the_ashen_sun", "kingdom_of_rust_and_ruin"]:
        researcher.generate_research_dossier(slug)
