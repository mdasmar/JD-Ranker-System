from __future__ import annotations

import argparse
import csv
import json
import math
import heapq
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


ROLE_TERMS = {
    "ml engineer",
    "machine learning engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "relevance engineer",
    "recommendation",
    "recommender",
    "retrieval",
    "vector search",
    "information retrieval",
    "nlp engineer",
    "ai engineer",
    "data scientist",
}

PRODUCT_TERMS = {
    "product company",
    "shipped",
    "production",
    "launched",
    "end-to-end",
    "real users",
    "scaled",
    "deployed",
}

STACK_TERMS = {
    "python": 1.6,
    "pytorch": 1.2,
    "tensorflow": 1.0,
    "embeddings": 2.2,
    "vector db": 2.4,
    "vector database": 2.4,
    "faiss": 2.0,
    "pinecone": 2.2,
    "weaviate": 2.2,
    "qdrant": 2.2,
    "milvus": 2.2,
    "opensearch": 2.0,
    "elasticsearch": 2.0,
    "ranking": 2.6,
    "learning to rank": 2.8,
    "ltr": 1.8,
    "retrieval": 2.6,
    "search": 1.5,
    "llm": 1.2,
    "rag": 1.6,
    "fine-tuning": 1.2,
    "lora": 1.0,
    "qlora": 1.0,
    "peft": 1.0,
    "hybrid search": 2.4,
    "embedding": 2.0,
}

WEAK_TERMS = {
    "marketing manager",
    "customer support",
    "operations manager",
    "consulting",
    "business analyst",
    "content writer",
    "graphic designer",
    "hr manager",
    "mechanical engineer",
}

INDIA_FOCUS_LOCATIONS = {
    "pune",
    "noida",
    "delhi",
    "delhi ncr",
    "gurugram",
    "gurgaon",
    "mumbai",
    "hyderabad",
}

RANGE = {
    "year_target_min": 5.0,
    "year_target_max": 9.0,
    "year_soft_min": 4.0,
    "year_soft_max": 12.0,
}

# Use the runtime date so recency scoring stays current.
TODAY = date.today()


@dataclass
class CandidateScore:
    candidate_id: str
    score: float
    reasoning: str


def normalize_text(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def count_hits(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def months_since(d: date | None) -> int | None:
    if d is None:
        return None
    return max(0, (TODAY.year - d.year) * 12 + TODAY.month - d.month)


def bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_location(candidate: dict[str, Any]) -> tuple[float, str]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    loc = normalize_text(profile.get("location", ""), profile.get("country", ""))
    current_mode = str(signals.get("preferred_work_mode", "")).lower()
    relocate = bool(signals.get("willing_to_relocate"))

    score = 0.0
    notes = []
    if any(place in loc for place in INDIA_FOCUS_LOCATIONS):
        score += 3.0
        notes.append("location fits India focus")
    elif profile.get("country", "").lower() == "india":
        score += 2.0
        notes.append("based in India")
    elif relocate:
        score += 1.0
        notes.append("willing to relocate")
    else:
        score -= 1.0
        notes.append("location less aligned")

    if current_mode in {"onsite", "hybrid", "flexible"}:
        score += 1.5
    if relocate:
        score += 0.75
    if not relocate and any(place in loc for place in {"bangalore", "bengaluru", "chennai", "kolkata", "austin", "toronto"}):
        score -= 0.75
    return score, ", ".join(notes)


def score_behavior(candidate: dict[str, Any]) -> tuple[float, str]:
    s = candidate["redrob_signals"]
    score = 0.0
    notes = []

    if s.get("open_to_work_flag"):
        score += 2.5
        notes.append("open to work")
    last_active = parse_date(s.get("last_active_date"))
    gap = months_since(last_active)
    if gap is not None:
        if gap <= 1:
            score += 3.0
            notes.append("very recent activity")
        elif gap <= 3:
            score += 2.0
            notes.append("recent activity")
        elif gap <= 6:
            score += 0.5
        else:
            score -= 2.5
            notes.append("inactive recently")
    if gap is not None and gap <= 2 and s.get("open_to_work_flag"):
        score += 0.5

    rr = float(s.get("recruiter_response_rate", 0.0))
    score += (rr - 0.30) * 5.0
    if rr >= 0.7:
        notes.append("strong recruiter response rate")
    elif rr <= 0.2:
        score -= 1.0
        notes.append("low recruiter response rate")

    notice = int(s.get("notice_period_days", 180))
    if notice <= 30:
        score += 2.0
        notes.append("short notice")
    elif notice <= 60:
        score += 1.0
    elif notice <= 90:
        score += 0.2
    else:
        score -= 1.5
        notes.append("long notice period")

    score += 0.8 if s.get("verified_email") else -0.5
    score += 0.8 if s.get("verified_phone") else -0.2
    score += 0.5 if s.get("linkedin_connected") else 0.0
    score += 0.02 * min(int(s.get("saved_by_recruiters_30d", 0)), 20)
    score += 0.005 * min(int(s.get("profile_views_received_30d", 0)), 100)
    score += 0.01 * min(int(s.get("interview_completion_rate", 0.0) * 100), 100)
    return score, ", ".join(notes)


def skill_score(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    profile = candidate["profile"]
    career_history = candidate.get("career_history", [])
    text = normalize_text(
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(job.get("title", "") + " " + job.get("description", "") for job in career_history),
        " ".join(skill.get("name", "") for skill in candidate.get("skills", [])),
    )
    career_text = normalize_text(
        " ".join(job.get("title", "") + " " + job.get("description", "") for job in career_history)
    )

    score = 0.0
    notes: list[str] = []
    term_hits = []
    for term, weight in STACK_TERMS.items():
        if term in text:
            score += min(weight, 2.5)
            term_hits.append(term)

    for skill in candidate.get("skills", []):
        name = str(skill.get("name", "")).lower()
        prof = str(skill.get("proficiency", "")).lower()
        endors = int(skill.get("endorsements", 0))
        dur = int(skill.get("duration_months", 0) or 0)
        if name in {"python", "pytorch", "tensorflow"}:
            score += 0.7
        if name in {"embeddings", "retrieval", "ranking", "search", "milvus", "pinecone", "weaviate", "qdrant", "faiss", "elasticsearch", "opensearch"}:
            score += 1.2
        score += min(1.5, endors / 20.0)
        score += min(1.2, dur / 24.0)
        if prof in {"advanced", "expert"}:
            score += 0.4

    weak_count = count_hits(text, WEAK_TERMS)
    if weak_count:
        score -= 1.4 * weak_count
        notes.append("contains unrelated career language")
    if count_hits(text, {"python", "ml", "machine learning", "retrieval", "ranking", "search", "embedding"}) >= 3 and count_hits(career_text, {"built", "shipped", "deployed", "production", "launched"}) == 0:
        score -= 2.0
        notes.append("skills are not grounded in delivery")
    if count_hits(text, {"senior ai engineer", "production ml systems", "hybrid retrieval system", "offline-online evaluation", "candidate-jd matching pipeline"}) >= 2 and len(career_history) <= 2:
        score -= 2.0
        notes.append("template-like seniority claims")
    if count_hits(career_text, {"search", "retrieval", "ranking", "recommendation"}) >= 2:
        score += 1.5
    if count_hits(career_text, {"vector", "embedding", "faiss", "elasticsearch", "opensearch", "pinecone", "milvus", "qdrant", "weaviate"}) >= 2:
        score += 1.0
    if "llm" in text or "rag" in text:
        if not any(term in text for term in {"product company", "deployed", "shipped", "production"}):
            score -= 1.5
            notes.append("AI keywords without production evidence")

    if term_hits:
        notes.append("matched " + ", ".join(term_hits[:4]))
    return score, notes


def experience_score(candidate: dict[str, Any]) -> tuple[float, str]:
    profile = candidate["profile"]
    years = float(profile.get("years_of_experience", 0.0))
    score = 0.0
    notes = []

    if RANGE["year_target_min"] <= years <= RANGE["year_target_max"]:
        score += 4.0
        notes.append("experience in target band")
    elif RANGE["year_soft_min"] <= years < RANGE["year_target_min"]:
        score += 2.0
        notes.append("slightly under target band")
    elif RANGE["year_target_max"] < years <= RANGE["year_soft_max"]:
        score += 1.5
        notes.append("senior but still plausible")
    else:
        score -= 1.0

    career = candidate.get("career_history", [])
    current_title = normalize_text(profile.get("current_title", ""))
    current_company = normalize_text(profile.get("current_company", ""))
    career_text = normalize_text(" ".join(j.get("title", "") + " " + j.get("description", "") + " " + j.get("company", "") for j in career))

    prod_terms = count_hits(career_text, PRODUCT_TERMS)
    if prod_terms:
        score += min(3.0, prod_terms * 0.6)
        notes.append("shows product/production delivery")
    if count_hits(career_text, {"built", "designed", "deployed", "shipped", "owned", "led"}) >= 3:
        score += 1.0

    role_hits = count_hits(career_text, ROLE_TERMS)
    if role_hits:
        score += min(3.5, role_hits * 0.9)
        notes.append("relevant technical trajectory")
    if any(term in current_title for term in {"ml engineer", "data scientist", "ai engineer", "search engineer", "applied scientist"}):
        score += 2.0
    elif any(term in current_title for term in {"backend", "software engineer", "full stack", "data engineer", "analytics engineer"}):
        score += 0.8
    elif any(term in current_title for term in WEAK_TERMS):
        score -= 2.5
        notes.append("current title is off-target")
    elif any(term in current_title for term in {"manager", "lead", "director", "head"}):
        score -= 1.5
        notes.append("title is managerial rather than hands-on")

    if current_company and any(x in current_company for x in {"google", "meta", "microsoft", "amazon", "inmobi", "zoho", "sharechat", "razorpay", "flipkart", "phonepe", "myntra", "unacademy", "swiggy", "curefit"}):
        score += 0.8

    summary = normalize_text(profile.get("summary", ""))
    if years < 4.0 and ("senior" in summary or "lead" in summary):
        score -= 1.5
        notes.append("senior language conflicts with experience")
    if years < 4.0 and count_hits(summary, {"search", "retrieval", "ranking", "recommendation"}) >= 2:
        score -= 0.75
    if years >= 5.0 and years <= 9.0 and count_hits(career_text, {"search", "retrieval", "ranking", "recommendation"}) >= 2:
        score += 1.0
    return score, ", ".join(notes)


def reasoning_for(candidate: dict[str, Any], parts: list[str]) -> str:
    profile = candidate["profile"]
    years = float(profile.get("years_of_experience", 0.0))
    title = profile.get("current_title", "")
    loc = profile.get("location", "")
    summary = profile.get("summary", "")
    summary_bits = []

    if any(t in normalize_text(title) for t in {"ml engineer", "ai engineer", "search engineer", "data scientist", "applied scientist"}):
        summary_bits.append(f"{title} background")
    elif any(t in normalize_text(summary) for t in {"retrieval", "ranking", "recommendation", "vector", "search"}):
        summary_bits.append("relevant search/retrieval experience")
    else:
        summary_bits.append(f"{title} with adjacent engineering experience")

    summary_bits.append(f"{years:.1f} yrs")
    if loc:
        summary_bits.append(loc)
    extras = [p for p in parts if p]
    if extras:
        summary_bits.append("; ".join(extras[:2]))
    return ". ".join(summary_bits).strip(".") + "."


def score_candidate(candidate: dict[str, Any]) -> CandidateScore:
    loc_score, loc_note = score_location(candidate)
    beh_score, beh_note = score_behavior(candidate)
    sk_score, sk_notes = skill_score(candidate)
    exp_score, exp_note = experience_score(candidate)

    raw = (
        1.00 * sk_score
        + 1.20 * exp_score
        + 1.15 * beh_score
        + 0.85 * loc_score
    )

    profile_text = normalize_text(
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        " ".join(j.get("description", "") for j in candidate.get("career_history", [])),
    )
    if count_hits(profile_text, {"openai", "chatgpt", "anthropic", "langchain"}) >= 2 and not any(
        term in profile_text for term in {"shipped", "production", "deployed", "real users"}
    ):
        raw -= 2.0
    if count_hits(profile_text, {"retrieval", "ranking", "recommendation", "search"}) >= 2 and count_hits(profile_text, {"built", "deployed", "owned", "led"}) >= 2:
        raw += 2.0

    score = round(raw, 4)

    parts = []
    parts.extend(sk_notes[:2])
    if exp_note:
        parts.append(exp_note)
    if loc_note:
        parts.append(loc_note)
    if beh_note:
        parts.append(beh_note)
    reasoning = reasoning_for(candidate, parts)

    return CandidateScore(candidate["candidate_id"], score, reasoning)


def iter_candidates(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or sample_candidates.json")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument("--limit", type=int, default=100, help="How many rows to emit; default 100")
    args = parser.parse_args()

    candidate_path = Path(args.candidates)
    sample_mode = candidate_path.suffix.lower() == ".json"

    if sample_mode:
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        scored = [score_candidate(c) for c in candidates]
        scored.sort(key=lambda x: (-x.score, x.candidate_id))
        top = scored[: args.limit]
    else:
        heap: list[tuple[float, str, CandidateScore]] = []
        for candidate in iter_candidates(candidate_path):
            scored = score_candidate(candidate)
            key = (scored.score, scored.candidate_id)
            if len(heap) < args.limit:
                heapq.heappush(heap, (key[0], key[1], scored))
            else:
                if key > (heap[0][0], heap[0][1]):
                    heapq.heapreplace(heap, (key[0], key[1], scored))
        top = [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, item in enumerate(top, start=1):
            writer.writerow([item.candidate_id, i, f"{item.score:.4f}", item.reasoning])


if __name__ == "__main__":
    main()
