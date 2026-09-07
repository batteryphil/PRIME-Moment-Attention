"""
PRIME-Scout: SQLite Persistence & Memory Vault
Stores tracked repositories, evaluation telemetry, sandbox execution logs, and daily digests.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from scout.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS repositories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_url TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        full_name TEXT NOT NULL,
        owner TEXT NOT NULL,
        description TEXT,
        stars INTEGER DEFAULT 0,
        forks INTEGER DEFAULT 0,
        language TEXT,
        topics TEXT,
        created_at TEXT,
        updated_at TEXT,
        discovered_at TEXT,
        status TEXT DEFAULT "DISCOVERED"
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id INTEGER NOT NULL,
        evaluated_at TEXT NOT NULL,
        viability_score INTEGER NOT NULL,
        alignment_score INTEGER NOT NULL,
        verdict TEXT NOT NULL,
        executive_pitch TEXT,
        technical_critique TEXT,
        synergy_notes TEXT,
        sandbox_status TEXT DEFAULT "SKIPPED",
        sandbox_log TEXT,
        FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS briefings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        briefing_date TEXT UNIQUE NOT NULL,
        report_path TEXT NOT NULL,
        summary TEXT,
        repo_count INTEGER DEFAULT 0,
        created_at TEXT
    );
    """)
    
    conn.commit()
    conn.close()

def upsert_repository(repo: Dict[str, Any]) -> int:
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    
    topics_json = json.dumps(repo.get("topics", []))
    
    cur.execute("""
    INSERT INTO repositories (repo_url, name, full_name, owner, description, stars, forks, language, topics, created_at, updated_at, discovered_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(repo_url) DO UPDATE SET
        stars=excluded.stars,
        forks=excluded.forks,
        description=excluded.description,
        updated_at=excluded.updated_at
    RETURNING id;
    """, (
        repo["repo_url"],
        repo.get("name", ""),
        repo.get("full_name", ""),
        repo.get("owner", ""),
        repo.get("description", ""),
        repo.get("stars", 0),
        repo.get("forks", 0),
        repo.get("language", ""),
        topics_json,
        repo.get("created_at", now_str),
        repo.get("updated_at", now_str),
        now_str,
        repo.get("status", "DISCOVERED")
    ))
    
    row = cur.fetchone()
    repo_id = row[0]
    conn.commit()
    conn.close()
    return repo_id

def record_evaluation(repo_id: int, eval_data: Dict[str, Any]) -> int:
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    
    cur.execute("""
    INSERT INTO evaluations (
        repo_id, evaluated_at, viability_score, alignment_score, verdict,
        executive_pitch, technical_critique, synergy_notes, sandbox_status, sandbox_log
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    RETURNING id;
    """, (
        repo_id,
        now_str,
        eval_data.get("viability_score", 50),
        eval_data.get("alignment_score", 50),
        eval_data.get("verdict", "MONITOR"),
        eval_data.get("executive_pitch", ""),
        eval_data.get("technical_critique", ""),
        eval_data.get("synergy_notes", ""),
        eval_data.get("sandbox_status", "SKIPPED"),
        eval_data.get("sandbox_log", "")
    ))
    
    eval_id = cur.fetchone()[0]
    
    # Update repo status
    cur.execute("UPDATE repositories SET status = 'EVALUATED' WHERE id = ?", (repo_id,))
    
    conn.commit()
    conn.close()
    return eval_id

def is_repo_evaluated(repo_url: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT e.id FROM evaluations e
    JOIN repositories r ON e.repo_id = r.id
    WHERE r.repo_url = ? LIMIT 1
    """, (repo_url,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def get_recent_evaluations(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT 
        r.id as repo_id, r.repo_url, r.name, r.full_name, r.owner, r.description,
        r.stars, r.forks, r.language, r.topics,
        e.id as eval_id, e.evaluated_at, e.viability_score, e.alignment_score,
        e.verdict, e.executive_pitch, e.technical_critique, e.synergy_notes,
        e.sandbox_status, e.sandbox_log
    FROM evaluations e
    JOIN repositories r ON e.repo_id = r.id
    ORDER BY e.id DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        d = dict(row)
        if d.get("topics"):
            try:
                d["topics"] = json.loads(d["topics"])
            except Exception:
                pass
        results.append(d)
    return results

def get_evaluation_by_url(repo_url: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT 
        r.id as repo_id, r.repo_url, r.name, r.full_name, r.owner, r.description,
        r.stars, r.forks, r.language, r.topics,
        e.id as eval_id, e.evaluated_at, e.viability_score, e.alignment_score,
        e.verdict, e.executive_pitch, e.technical_critique, e.synergy_notes,
        e.sandbox_status, e.sandbox_log
    FROM evaluations e
    JOIN repositories r ON e.repo_id = r.id
    WHERE r.repo_url = ?
    ORDER BY e.id DESC
    LIMIT 1
    """, (repo_url,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if d.get("topics"):
        try:
            d["topics"] = json.loads(d["topics"])
        except Exception:
            pass
    return d

def record_briefing(date_str: str, report_path: str, summary: str, repo_count: int) -> int:
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    cur.execute("""
    INSERT INTO briefings (briefing_date, report_path, summary, repo_count, created_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(briefing_date) DO UPDATE SET
        report_path=excluded.report_path,
        summary=excluded.summary,
        repo_count=excluded.repo_count
    RETURNING id;
    """, (date_str, report_path, summary, repo_count, now_str))
    briefing_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return briefing_id

def get_briefings(limit: int = 15) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT id, briefing_date, report_path, summary, repo_count, created_at
    FROM briefings
    ORDER BY briefing_date DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("[+] Database initialized successfully at:", DB_PATH)
