# Baseline Branch

This branch contains only the code, data, and pipeline for the baseline sentiment analysis model.

- All code is in `src/` (models, data processing).
- Data is tracked with DVC and stored in `data/`.
- Only baseline-relevant scripts and configs are present.
- For experiment tracking, connect to MLflow and DagsHub as described in the main branch.

To reproduce the baseline pipeline:
```sh
dvc repro
```
```

---

### 2. **Remove Any Unused or Empty Files**

- If there are any empty or placeholder files (like an empty README), update or remove them as appropriate.

---

### 3. **Check .gitignore**

Ensure `.gitignore` is present and includes: