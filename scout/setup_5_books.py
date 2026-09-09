"""
Setup and outline all 5 books for 'Beyond the Event Horizon'.
"""

import json
import sqlite3
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
NOVELS_ROOT = PROJECT_ROOT / "novels"
VAULT_DB_PATH = PROJECT_ROOT / "scout" / "vault" / "scout_vault.db"

BOOKS_DATA = [
    {
        "book_number": 1,
        "slug": "beyond_the_event_horizon",
        "title": "Relic of the Void",
        "synopsis": "An experimental test pilot loses control near a rotating black hole, slingshotting around the event horizon under extreme gravitational time dilation. Emerging 1,000 years into the future, he is salvaged by a fierce pirate captain, navigating vacuum warfare and shocking truths about humanity's lost home."
    },
    {
        "book_number": 2,
        "slug": "beyond_the_event_horizon_book_2",
        "title": "Corsair's Nebula",
        "synopsis": "Taking refuge inside the lawless Asteroid Belt Syndicates of Ostra-9, Leo Mercer proves his analog flight instincts in deadly close-quarters dogfights among spinning mega-asteroids. As his romantic bond with Captain Astrid 'Vex' Ross deepens into fierce loyalty, they discover an encrypted Golden Age military beacon pulsing from the heart of an ionized plasma nebula."
    },
    {
        "book_number": 3,
        "slug": "beyond_the_event_horizon_book_3",
        "title": "The Star-Eater's Wake",
        "synopsis": "A perilous deep-space expedition through uncharted stellar nurseries and blinding magnetar storms. Leo and Astrid witness the sublime majesty and terror of dying stars while evading the relentless pursuit of Commodore Hawke's Dominion fleet. Amidst catastrophic radiation leaks and high-g drift maneuvers, feelings unspoken erupt into passionate devotion."
    },
    {
        "book_number": 4,
        "slug": "beyond_the_event_horizon_book_4",
        "title": "Sovereigns of the Dark",
        "synopsis": "Open war breaks out between the autocratic Archon Dominion and the united corsair flotillas. In a devastating ambush, Astrid's flagship is crippled and she is captured by Commodore Hawke. Refusing to lose the woman he loves, Leo takes command of an archaic heavy strike craft and mounts an impossible high-g rescue mission into the heart of the Dominion's orbital fortress."
    },
    {
        "book_number": 5,
        "slug": "beyond_the_event_horizon_book_5",
        "title": "The Cradle of Stars",
        "synopsis": "The epic series finale. Armed with decoded pre-collapse navigation keys, Leo and Astrid lead the free fleet on a return voyage to the forbidden Sol system. They uncover the ancient truth behind humanity's fall and execute one final, daring relativistic slingshot maneuver around Jupiter's gravity well to break the Dominion's ultimate super-dreadnought, inaugurating a new golden dawn of freedom."
    }
]

# Chapter blueprints for Books 2 to 5
BOOK_CHAPTERS_MAP = {
    2: [
        ("The Ghost of Ostra-9", "Docking the battered Starlight Marauder inside the cavernous hollowed-out pirate asteroid."),
        ("Bounty on an Ancient Ghost", "Word leaks of a living pre-collapse pilot; syndicates and bounty hunters prowl the pressurized bazaars."),
        ("Neon and Kinetic Fire", "A shootout through the hydroponics bay and mag-train tunnels; Astrid and Leo fight back-to-back."),
        ("The Scrap Market Barter", "Leo helps Orlo retrofit the Marauder using archaic propulsion bypasses; friction turns into mutual respect."),
        ("A Drink in the Smuggler's Ring", "Astrid and Leo share a rare quiet moment at an orbital speakeasy; simmering romantic vulnerability."),
        ("Shadows of the Syndicate", "The Syndicate Boss Demetrius demands Leo as payment for docking rights; Astrid refuses with drawn steel."),
        ("The Hangar Bay Escape", "A daring midnight breakout; Leo manually pilots the Marauder through narrow asteroid docking tubes under fire."),
        ("Into the Ion Veil", "Fleeing into the blinding electromagnetic storms of the Veil Nebula; radiation screens humming."),
        ("The Thermal Ambush", "Two corsair wolf-pack interceptors attack from radar shadows; vacuum railgun duel."),
        ("Manual Override", "When automated targeting fails in the ion storm, Leo uses analog drift optics to score two direct hits."),
        ("The Wreckage of Valkyrie", "Discovering a derelict pre-collapse military dreadnought drifting inside the nebula."),
        ("Boarding the Ghost Fleet", "Spacewalk across the silent void into the derelict's shattered spine; zero-G exploration."),
        ("The Sealed Command Vault", "Leo uses his thumbprint and DNA to open the command vault; ancient emergency beacons flare."),
        ("Whispers from the Golden Age", "Decrypting the dreadnought's black box log; chilling hints about why Sol was quarantined."),
        ("Breached Containment", "An unstable micro-fusion core begins a prompt-critical runaway; harrowing race back to the airlock."),
        ("Bruises and Starlight", "Astrid bandages Leo's burns in her cabin; intense physical proximity breaks their emotional barriers."),
        ("The First Touch", "A quiet, breathless night in the captain's quarters; confessing what they mean to each other under the nebula's glow."),
        ("The Picket Hunter Arrives", "A Dominion stealth frigate tracks the dreadnought's beacon; torpedoes in the dark."),
        ("The Nebula Slingshot", "Using a protostar's gravity well to sling the Marauder past the hunter's firing arc."),
        ("The Beacon's Promise", "Transmitting the coordinates of the Star-Eater Nebula; setting course for the unknown.")
    ],
    3: [
        ("Edge of the Abyss", "Entering the forbidden Star-Eater sector; gravitational gradients and glowing purple plasma."),
        ("The Magnetar's Pulse", "A nearby magnetar emits periodic gamma-ray bursts; precision orbital timing required."),
        ("The Silent Cathedrals", "Encountering crystal-dense comet rings; breathtaking cosmic majesty and silence."),
        ("Radiation Lockout", "Shield failure forces the crew into the heavily leaded storm shelter; forced intimacy in cramped quarters."),
        ("Stolen Breaths", "Leo and Astrid confront the reality of their mortality; passionate embrace amidst alarm lights."),
        ("The Ghost Signal Decoded", "Orlo deciphers the beacon: an ancient automated gateway leading toward Sol's perimeter."),
        ("The Corsair Council", "Astrid calls a holographic council of pirate captains; warning them of Dominion mobilization."),
        ("Betrayal at the Coordinates", "A traitor captain leaks their rendezvous point to Commodore Hawke."),
        ("The Trap Springs", "Dominion cruisers jump out of sub-light, surrounding the small fleet."),
        ("Silent Fleet Warfare", "Massive railgun barrages, PDCs lighting the void, silent thermonuclear torpedo detonations."),
        ("The Marauder's Stand", "Astrid maneuvers between two capital ships, using their own crossfire against them."),
        ("Leo's High-G Burn", "Taking the secondary flight helm, Leo pulls an 11-G vector change to rescue a friendly corvette."),
        ("Armor Pierced", "A kinetic slug shears through the Marauder's forward radiator wing; fire in the starboard nacelle."),
        ("Spacewalk Under Fire", "Leo and Orlo suit up to manually eject the burning radiator before it cooks the fusion reactor."),
        ("The Magnetar Flare", "A massive pulse from the magnetar blinds all radar and thermal sensors across the battle sphere."),
        ("Blind Navigation", "Leo flies blind using inertial dead-reckoning, slipping the Marauder into the magnetar's shadow."),
        ("Casualties of the Void", "Tending to the wounded in the cargo deck; Astrid's grief and Leo's comforting touch."),
        ("Vows in the Dark", "Astrid and Leo acknowledge that they are inseparable; a promise to see Earth together."),
        ("The Gateway Discovered", "The ancient pre-collapse Jump Ring stands dormant before them, wrapped in cosmic dust."),
        ("The Key Turned", "Activating the gateway; the ring pulses with sapphire light as the Dominion closes in.")
    ],
    4: [
        ("The Gateway Transit", "The violent spacetime transition through the jump ring; emerging on the doorstep of the Core Systems."),
        ("The Dominions Iron Grip", "Witnessing the heavily militarized Dominion fortress world of Kaelus-Prime."),
        ("Scavenger Infiltration", "Disguising the Marauder as an ore transport to slip past orbital security nets."),
        ("The Ambush at Gate Three", "Commodore Hawke was waiting; automated tractor nets and EMP minefields ensnare the Marauder."),
        ("The Last Stand on the Bridge", "Astrid orders the crew into the escape lifepods while she holds the bridge controls."),
        ("Captured", "Astrid is taken prisoner by Hawke; the Marauder is boarded and stripped."),
        ("The Pilot in the Shadows", "Leo, hiding in a hidden crawlspace with Orlo, evades capture as the ship is towed to the prison station."),
        ("The Iron Spire", "Astrid faces interrogation by Hawke, refusing to disclose the coordinates of the Sol beacon."),
        ("Planning the Impossible", "Leo and Orlo recruit rogue dockworkers and imprisoned smugglers on the station."),
        ("Stealing a Relic", "Leo infiltrates the imperial impound bay and hot-wires an experimental heavy strike fighter."),
        ("Blackout at Sector 7", "Orlo detonates an EMP charge in the station's primary cooling conduit; emergency lights only."),
        ("Gunfight on the Gantry", "Leo fights through Dominion storm-troopers, armed with a high-velocity rail carbine."),
        ("Breaching the High Security Block", "Leo blows the cell doors; reunited with Astrid in an emotional, breathless embrace."),
        ("No Time for Tears", "Fighting their way back to the strike craft under heavy fire from gun emplacements."),
        ("The Launch Bay Ejection", "Launching out of the depressurizing hangar bay into a sea of patrolling gunboats."),
        ("Atmospheric Skim", "Leo pilots the craft into the upper atmosphere of Kaelus to shake missile locks."),
        ("The Reunion on the Corsair Flagship", "Meeting up with the regrouped pirate fleet; Astrid takes command of the allied armada."),
        ("Declaration of War", "Astrid broadcasts a call to arms across all free systems: the march on Sol begins."),
        ("The Fleet Gathers", "Hundreds of modified corvettes, mining rigs, and corsair cruisers form up in the void."),
        ("Jump to Sol", "The allied armada initiates the coordinated burn toward humanity's quarantined origin.")
    ],
    5: [
        ("The Dead Star System", "Emerging in the Sol system; the eerie, silent cemetery of humanity's ancient expansion."),
        ("Ruins of the Orbital Ring", "Passing through the shattered remnants of Mars's orbital elevator and docking rings."),
        ("The Truth of the Collapse", "Recovering the orbital archives of Luna: the Dominion was born from the military junta that quarantined Earth."),
        ("The Earth Below", "Leo gazes down upon his home planet for the first time in 1,000 years: oceans green and wild, scars of old wars healed."),
        ("Hawke's Grand Armada", "The Dominion's supreme dreadnought 'The Archon-Imperator' arrives with its battlefleet."),
        ("The Battle for Humanity's Cradle", "The largest vacuum fleet battle in history commences between Earth and the Moon."),
        ("Railgun Salvos and Plasma Storms", "Capital ships colliding; silent detonations turning the blackness into midday."),
        ("The Marauder's Sacrifice", "The Starlight Marauder takes critical damage protecting the allied fleet's carrier."),
        ("Transfer to Chronos-1", "Leo's original ship, refitted by Orlo with modern ordnance, becomes their last spearhead."),
        ("Astrid and Leo in the Cockpit", "Flying together in the tandem cockpit of the Chronos-1; complete synchronization."),
        ("The Jovian Maneuver", "Hawke retreats toward Jupiter's massive radiation belts, daring them to follow."),
        ("Into the Radiation Belts", "High-radiation transit past Io and Europa; shields screaming at the limit."),
        ("The Hyperbolic Slingshot", "Leo proposes one final relativistic slingshot around Jupiter's gravity well to out-speed the dreadnought."),
        ("Crushing G-Forces", "14-G deceleration burn; holding hands through the agony of the gravitational plunge."),
        ("The Blind Flank", "Emerging from Jupiter's shadow at 0.15c, completely bypassing the dreadnought's forward shields."),
        ("Spinal Kill", "Chronos-1 fires an antimatter-tipped kinetic penetrator directly into the dreadnought's fusion core."),
        ("The Fall of the Archon", "The dreadnought shatters in a blinding silent nova; Dominion forces surrender."),
        ("Setting Foot on Earth", "Leo and Astrid's shuttle touches down on a lush, wild beach on the coast of ancient North America."),
        ("The Smell of Rain", "Leo steps out into real air, real wind, and real sunlight; sharing the first breath with Astrid."),
        ("A New Dawn", "Founding a new, free colony under the stars where freedom and science thrive together.")
    ]
}

def setup_all_books():
    conn = sqlite3.connect(VAULT_DB_PATH)
    cur = conn.cursor()

    for b in BOOKS_DATA:
        b_num = b["book_number"]
        slug = b["slug"]
        title = b["title"]
        synopsis = b["synopsis"]
        book_dir = NOVELS_ROOT / slug
        book_dir.mkdir(parents=True, exist_ok=True)
        chapters_dir = book_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        export_dir = book_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        bible_path = book_dir / "BOOK_BIBLE.json"
        
        # Build chapter outline
        if b_num == 1 and bible_path.exists():
            print(f"[✓] Book 1 already has active BOOK_BIBLE.json")
        else:
            ch_list = []
            ch_data = BOOK_CHAPTERS_MAP.get(b_num, [])
            for ch_idx in range(1, 21):
                if ch_idx <= len(ch_data):
                    ch_title, ch_sum = ch_data[ch_idx - 1]
                else:
                    ch_title, ch_sum = f"Chapter {ch_idx}", f"Tactical space opera chapter continuing the mission of {title}."
                ch_list.append({
                    "chapter": ch_idx,
                    "title": ch_title,
                    "target_words": 2500,
                    "actual_words": 0,
                    "act": 1 if ch_idx <= 5 else (2 if ch_idx <= 15 else 3),
                    "status": "QUEUED",
                    "summary": ch_sum
                })

            book_bible = {
                "slug": slug,
                "title": title,
                "series_title": "Beyond the Event Horizon",
                "book_number": b_num,
                "subgenre": "Hard Science Fiction Space Opera",
                "target_words": 50000,
                "total_chapters": 20,
                "heat_level": 3,
                "synopsis": synopsis,
                "chapters": ch_list
            }
            bible_path.write_text(json.dumps(book_bible, indent=2), encoding="utf-8")
            print(f"[✓] Initialized BOOK_BIBLE.json for Book {b_num}: '{title}' ({slug})")

        # Upsert into novel_library
        cur.execute("""
            INSERT OR REPLACE INTO novel_library 
            (slug, title, subgenre, target_words, current_words, total_chapters, completed_chapters, status, heat_level, synopsis, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        """, (
            slug,
            f"Beyond the Event Horizon: {title}",
            "Hard Science Fiction Space Opera",
            50000,
            4599 if b_num == 1 else 0,
            20,
            2 if b_num == 1 else 0,
            "WRITING" if b_num == 1 else "QUEUED",
            3,
            synopsis
        ))

    conn.commit()
    conn.close()
    print("[★] All 5 books in 'Beyond the Event Horizon' are outlined, registered, and ready for autonomous authoring!")

if __name__ == "__main__":
    setup_all_books()
