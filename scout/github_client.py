"""
PRIME-Scout: GitHub Discovery & Code Fetcher
Searches GitHub API for trending and keyword-matched repositories aligned with
efficient attention, ROCm kernels, continuous memory, and sub-quadratic architectures.
Includes offline high-relevance seed repositories for guaranteed operation.
"""

import os
import time
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from scout.config import RESEARCH_PROFILE

# Curated high-relevance repos used as seeds or when API rate limits apply
SEED_REPOSITORIES = [
    {
        "repo_url": "https://github.com/sustcsonglin/flash-linear-attention",
        "name": "flash-linear-attention",
        "full_name": "sustcsonglin/flash-linear-attention",
        "owner": "sustcsonglin",
        "description": "Efficient hardware-accelerated linear attention mechanisms including RetNet, GLA, RWKV, Mamba, and DeltaNet in Triton.",
        "stars": 2450,
        "forks": 280,
        "language": "Python",
        "topics": ["linear-attention", "triton", "transformer", "recurrent", "llm"]
    },
    {
        "repo_url": "https://github.com/state-spaces/mamba",
        "name": "mamba",
        "full_name": "state-spaces/mamba",
        "owner": "state-spaces",
        "description": "Linear-Time Sequence Modeling with Selective State Spaces.",
        "stars": 13800,
        "forks": 1400,
        "language": "Python",
        "topics": ["state-space-models", "linear-attention", "deep-learning", "cuda", "rocm"]
    },
    {
        "repo_url": "https://github.com/lucidrains/recurrent-memory-transformer-pytorch",
        "name": "recurrent-memory-transformer-pytorch",
        "full_name": "lucidrains/recurrent-memory-transformer-pytorch",
        "owner": "lucidrains",
        "description": "Implementation of Recurrent Memory Transformer in PyTorch for scaling transformers to 1M+ tokens.",
        "stars": 820,
        "forks": 95,
        "language": "Python",
        "topics": ["recurrent-attention", "long-context", "transformers", "pytorch"]
    },
    {
        "repo_url": "https://github.com/vllm-project/vllm",
        "name": "vllm",
        "full_name": "vllm-project/vllm",
        "owner": "vllm-project",
        "description": "A high-throughput and memory-efficient inference and serving engine for LLMs with PagedAttention and ROCm support.",
        "stars": 32000,
        "forks": 5100,
        "language": "Python",
        "topics": ["llm-serving", "paged-attention", "rocm", "cuda", "vllm"]
    },
    {
        "repo_url": "https://github.com/ROCm/triton",
        "name": "triton",
        "full_name": "ROCm/triton",
        "owner": "ROCm",
        "description": "Development repository for the Triton backend targeting AMD ROCm GPUs.",
        "stars": 1100,
        "forks": 240,
        "language": "C++",
        "topics": ["rocm", "hip", "triton-compiler", "gpu-kernels"]
    },
]

class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.headers = {
            "User-Agent": "PRIME-Scout-Agent/1.0",
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def _api_get(self, url: str) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"[!] GitHub API rate limit reached (HTTP 403) on {url}")
            else:
                print(f"[!] GitHub API HTTP {e.code}: {e.reason} on {url}")
        except Exception as e:
            print(f"[!] GitHub API request failed: {e}")
        return None

    def search_repositories(self, query: str, sort: str = "stars", limit: int = 10) -> List[Dict[str, Any]]:
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort={sort}&order=desc&per_page={limit}"
        data = self._api_get(url)
        if not data or "items" not in data:
            return []
        
        repos = []
        for item in data["items"]:
            repos.append({
                "repo_url": item.get("html_url"),
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "owner": item.get("owner", {}).get("login"),
                "description": item.get("description") or "",
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "language": item.get("language") or "Python",
                "topics": item.get("topics", []),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
            })
        return repos

    def fetch_readme(self, full_name: str) -> str:
        url = f"https://api.github.com/repos/{full_name}/readme"
        data = self._api_get(url)
        if data and "download_url" in data:
            try:
                req = urllib.request.Request(data["download_url"], headers={"User-Agent": "PRIME-Scout-Agent/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                print(f"[!] Failed to download README for {full_name}: {e}")
        
        # Fallback raw github url
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/{full_name}/{branch}/README.md"
            try:
                req = urllib.request.Request(raw_url, headers={"User-Agent": "PRIME-Scout-Agent/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except Exception:
                pass
        return "No README available."

    def discover_candidates(self, max_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Discovers candidate repositories matching the user research profile.
        Combines live GitHub search with high-relevance seed repositories.
        """
        candidates = {}
        
        # Add seeds first
        for seed in SEED_REPOSITORIES:
            candidates[seed["repo_url"]] = seed

        # Query GitHub API for top queries
        for query in RESEARCH_PROFILE["search_queries"][:4]:
            try:
                found = self.search_repositories(query, limit=max_per_query)
                for r in found:
                    candidates[r["repo_url"]] = r
                time.sleep(0.5) # Gentle rate limiting
            except Exception as e:
                print(f"[!] Query failed for {query}: {e}")
                
        return list(candidates.values())

    def calculate_alignment_heuristic(self, repo: Dict[str, Any]) -> float:
        """
        Fast preliminary heuristic scoring (0-100) based on keyword overlap
        in repo description, name, topics.
        """
        score = 20.0
        text = f"{repo.get("name", "")} {repo.get("description", "")} {" ".join(repo.get("topics", []))}".lower()
        
        for kw in RESEARCH_PROFILE["relevance_keywords"]:
            if kw.lower() in text:
                score += 8.0
                
        # Bonus for python or c++ or rocm
        lang = str(repo.get("language", "")).lower()
        if lang in ["python", "c++", "triton", "cuda"]:
            score += 10.0
            
        return min(100.0, score)

if __name__ == "__main__":
    client = GitHubClient()
    print("[*] Testing GitHubClient discovery...")
    candidates = client.discover_candidates(max_per_query=2)
    print(f"[+] Found {len(candidates)} candidate repositories:")
    for c in candidates[:5]:
        score = client.calculate_alignment_heuristic(c)
        print(f"    - [{c["stars"]}★ | Heuristic: {score:.0f}] {c["full_name"]}: {c["description"][:60]}...")
