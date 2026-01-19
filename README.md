# Description

My thought was to try out http 2 or 3 and to see differences. TBA

# Some initial theory of http 1, 2 and 3
TBA some diagrams

# How the test was done
Ideal cases

## Scenarios

### Many small ones

### Mimic real page load

### Large file

Why only 1 concurency?



```bash
python bench_run.py bench/scenario_many_small.yaml \                                       
  --curl "$(brew --prefix curl)/bin/curl" \
  --out out/good_many_small.jsonl
```

```bash
python bench_run.py bench/scenario_real_page.yaml \                                        
  --curl "$(brew --prefix curl)/bin/curl" \
  --out out/good_real_page.jsonl
```

```bash
python bench_run.py bench/scenario_large_file.yaml \
  --curl "$(brew --prefix curl)/bin/curl" \
  --out out/good_large_file.jsonl
```

TBA

Network latency


# Results   



# Setting up

* Install Caddy https://caddyserver.com/docs/install
```
brew install caddy
```

### Misc



* Add `export PATH="$(brew --prefix curl)/bin:$PATH"` to ~/.zshrc (|#TODO add automatic instruction)

## Running bench_run.py locally

**Dependencies**
- Python 3.8+ with `venv`
- `curl` built with h1/h2/h3 support (`brew install curl` on macOS and ensure it is first on `PATH`)
For Mac (ARM chip) I needed to point to newer curl version (the one which comes preinstalled doesn't support http 3)
* 
```
brew install curl
```

**Setup**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pyyaml
```

**Run**
Use any scenario file under `bench/` (YAML or JSON). Results are appended to the JSONL file you specify.
```bash
python bench_run.py bench/scenario_many_small.yaml --out out/results.jsonl
```
The script writes a metadata line plus one result per request to `out/results.jsonl`. Adjust `--reps` and `--concurrency` to override scenario defaults.
