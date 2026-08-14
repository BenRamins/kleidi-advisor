# results/

`kleidi-advisor bench` writes one JSON file per run to this directory, named `<stem>-<tag>.json`.
`kleidi-advisor report` reads every `*.json` file here (skipping any without `"schema": 1`) and
renders the results table, headline, and plot. These files are machine-generated and gitignored
except for this README — the box day fills this directory in, it does not ship pre-populated.

## Schema (`schema: 1`)

```json
{
  "schema": 1,
  "model": "llama-3.1-8b-q4_k_m.gguf",
  "tag": "baseline",
  "threads": 16,
  "instance": "Azure Standard_E8ps_v6 (Cobalt 100, Neoverse N2), 8 threads",
  "llama_cpp_commit": "1692f9e50",
  "timestamp_utc": "2026-08-14T09:12:33Z",
  "argv": ["llama-bench", "-m", "...", "-p", "512", "-n", "128", "-r", "5", "-o", "json"],
  "metrics": {
    "pp512": {"runs": [413.9, 415.1], "median": 414.5, "stdev": 0.6, "unit": "tok/s"},
    "tg128": {"runs": [28.1, 28.5], "median": 28.3, "stdev": 0.2, "unit": "tok/s"}
  },
  "ppl": {"value": 6.7841, "corpus": "wikitext-2-raw", "chunks": null}
}
```

`ppl` is `null` when not measured. `instance` and `llama_cpp_commit` are `null` until supplied —
`--instance` sets the first; the second is recorded by hand from `git rev-parse HEAD` in the
llama.cpp checkout. `report` omits an instance label rather than inventing one.
