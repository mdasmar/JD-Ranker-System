# Redrob Ranker Sandbox

This repository contains a CPU-only ranking solution for the Redrob hackathon challenge. It takes a candidate pool, scores each profile against the job description, and writes a top-100 CSV submission with short reasoning for each row.

## What is in this repo

- `rank.py` - main ranking script that reads `candidates.jsonl` and writes `submission.csv`
- `app.py` - Streamlit sandbox for running the ranker on sample input
- `validate_submission.py` - CSV format validator for the challenge rules
- `submission.csv` - the generated submission file
- `submission_metadata.yaml` - submission metadata for the portal
- `sample_candidates.json` - small sample candidate set for local testing
- `requirements.txt` - pinned Python dependency list
- `data/` - challenge reference files, schema, docs, and sample submission

## What should stay out of GitHub

- `venv/`
- `__pycache__/`
- `candidates.jsonl` because it is very large
- temporary validation outputs
- editor, OS, and backup files

## Repo layout

```text
.
|-- app.py
|-- rank.py
|-- validate_submission.py
|-- submission.csv
|-- submission_metadata.yaml
|-- sample_candidates.json
|-- requirements.txt
|-- README.md
|-- data/
|   |-- candidate_schema.json
|   |-- job_description.docx
|   |-- redrob_signals_doc.docx
|   |-- submission_spec.docx
|   |-- sample_submission.csv
|   |-- submission_metadata_template.yaml
|   `-- README.docx
```

## Local setup

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce the submission

```powershell
python rank.py --candidates candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

## Run the sandbox

```powershell
streamlit run app.py
```
sandbox_link: "https://jd-ranker-system-app.streamlit.app/"

## Notes

- The ranker is rule-based and CPU-only.
- The `score` column is a sortable float and does not need to be normalized to `0..1`.
- Recency scoring uses the runtime date so the ranking stays current.
- The sample uploader in `app.py` accepts a JSON candidate array and falls back to `sample_candidates.json` in the repo root.
