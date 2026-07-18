# Vendored copy — provenance

- **Source:** https://github.com/hyunsungkim-ds/elastic
- **Commit:** bc41bcdf43451ae639c6ae7b299c1ccd3712d00e (2025-11-04)
- **Licence:** MPL-2.0 (see `LICENSE`)
- **Subset:** `sync/` (the synchronization/detection algorithm), `tools/match_data.py`
  (the speed/acceleration signal preprocessing), upstream `README.md`,
  `requirements.txt`. Omitted: `docs/` media, notebooks, dataset tooling.
- **Local modifications:** none. This copy is **reference-only** — it is not
  imported anywhere at runtime. The production detectors are an independent
  numpy port in `src/geometry/key_moments.py` + `src/geometry/signal.py`
  (see `third_party/README.md` for why).
