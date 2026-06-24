import csv
import io
import json
from pathlib import Path

import streamlit as st

from rank import score_candidate


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample_candidates.json"


def load_candidates(uploaded_file):
    if uploaded_file is not None:
        raw = uploaded_file.read().decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def rank_candidates(candidates):
    scored = [score_candidate(c) for c in candidates]
    scored.sort(key=lambda x: (-x.score, x.candidate_id))
    rows = []
    for rank, item in enumerate(scored[:100], start=1):
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "rank": rank,
                "score": f"{item.score:.4f}",
                "reasoning": item.reasoning,
            }
        )
    return rows


st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Ranker Sandbox")
st.caption("Upload any sample candidate file or click Run ranking to use existing sample_candidates for ranking.")

uploaded = st.file_uploader("Upload sample_candidates.json", type=["json"])

if st.button("Run ranking"):
    candidates = load_candidates(uploaded)
    if not candidates:
        st.error("No candidates found in the input.")
    else:
        rows = rank_candidates(candidates)
        st.success(f"Ranked {len(candidates)} candidates and produced top {len(rows)}.")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Download CSV",
            data=buffer.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
        )
