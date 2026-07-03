**English** | [日本語](README.ja.md)

# engram — Human-like memory for AI agents (MCP server)

Persistent memory shared by Claude Code, Codex, and Antigravity (Gemini CLI).
The more a memory is used, the easier it is to recall; unused memories sink
but never disappear — the same dynamics as human memory.

> Note: the setup wizard and CLI messages are currently in Japanese, and the
> default embedding model (Ruri-v3) is Japanese-focused. The engine itself is
> language-agnostic; you can switch models with `embed_model` in
> `~/.engram/config.toml` (re-run `engram reindex` after changing it).

---

## Quick start

### Option 1: one-line install

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.ps1 | iex
```

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.sh | sh
```

This single line installs uv, installs engram, and runs the setup wizard.
(macOS: git is required — run `xcode-select --install` first if you don't
have it.)

### Option 2: manual install in three commands

Windows (PowerShell):

```powershell
# 1. Install uv (skip if you already have it)
irm https://astral.sh/uv/install.ps1 | iex

# 2. Install engram
uv tool install --python 3.12 git+https://github.com/ricoaiproject-cmd/engram.git

# 3. Run the setup wizard
engram setup
```

macOS / Linux:

```bash
# 1. Install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install engram (force a uv-managed Python — see note below)
UV_PYTHON_PREFERENCE=only-managed uv tool install --python 3.12 git+https://github.com/ricoaiproject-cmd/engram.git

# 3. Run the setup wizard
engram setup
```

> Why a uv-managed Python? engram needs a Python whose SQLite supports
> loadable extensions (for sqlite-vec). uv-managed Python provides this;
> system / python.org builds on macOS do not, and uv would otherwise prefer
> them when present — hence `UV_PYTHON_PREFERENCE=only-managed` (install.sh
> sets it for you). `engram doctor` has a check row for this.

The setup wizard automatically:
- creates the config file (`~/.engram/config.toml`)
- initializes the memory folder
- downloads the embedding model (first run only, ~500 MB)
- registers engram with Claude Code / Codex / Antigravity
- registers the hooks for auto-encoding and proactive recall (Claude Code)

---

## After installation

### Just talk to your agent

Your agent performs every engram operation on its own. You simply have normal
conversations, and memories accumulate and get used automatically.

### Take the onboarding interview first (recommended)

Ask your agent:

```
Read ~/.engram/ONBOARDING.md and interview me.
```

Seeding your working style, preferences, and background makes the memory
useful from day one.

### Check your environment

```powershell
engram doctor
```

Shows Python version, config file, model cache, embedding backend
(ONNX / torch), install health (detects leftover `~ngram`-style remnants of a
failed pip reinstall that break `import engram`), per-agent registration
status as `[OK]` / `[NG]` / `[--]`, FTS5 availability (whether SQLite's
full-text search extension is loaded, since keyword search silently degrades
without it), and a perf summary section that surfaces recent MCP tool-call
and startup timings recorded to `data_dir/perf/perf_log.jsonl` (see below) so
a "something feels slow" complaint can be diagnosed from data rather than
guesswork.

### Re-run setup (e.g. after installing a new agent)

```powershell
engram setup
```

Safe to run any number of times (idempotent). Only unregistered agents are
added.

### Choose which agents to register

Even with multiple agents installed, you can connect engram to just the ones
you want.

```powershell
# Register with Claude Code only
engram setup --agents claude

# Register with both Claude Code and Codex
engram setup --agents claude,codex
```

Valid names: `claude` / `codex` / `gemini` (`antigravity` is an alias for
`gemini`). Without `--agents`, interactive mode lists the detected agents and
lets you pick by number (Enter selects all); `--non-interactive` registers all
detected agents as before.

### Faster startup with ONNX (new in v0.6)

Run this once:

```powershell
engram export-onnx
```

This converts the embedding model to ONNX (no extra dependencies — the
already-installed torch does the one-time conversion) and the server starts
using it automatically (`embed_backend=auto`). Startup drops from 12–24 s
(torch import) to **~2 s**, and no MCP client timeout tuning is needed
anymore.

Safety: the export verifies that the ONNX embeddings match the torch path on
a set of sample texts (min cosine ≥ 0.999, including a long text that crosses
ModernBERT's sliding-window boundary) and refuses to install a drifted model —
a silently drifted embedding space would corrupt recall against your existing
`index.db`.

`embed_backend` in `config.toml` (or `ENGRAM_EMBED_BACKEND`) selects the
runtime: `auto` (default; ONNX if exported, else torch), `onnx` (forced;
errors if not exported), `torch` (forced fallback).

### Startup mode (`ENGRAM_PRELOAD`)

With the ONNX backend the preload is cheap and the default `blocking` needs no
tuning. The numbers below describe the **torch fallback** (no ONNX model
exported yet) — kept because they explain why the design looks like this. The
`ENGRAM_PRELOAD` environment variable controls when the model load happens:

| Value | Behavior |
|---|---|
| `blocking` (default) | Load the model on the main thread before answering the handshake (~12–24 s warm, 50+ s cold). Every `recall` after connect responds instantly. Raise your client's MCP startup timeout to 120 s or more (for Claude Code: `MCP_TIMEOUT=120000`). |
| `background` | Respond to the handshake immediately and load the model in a background thread. **Caution:** on Windows, importing torch on a non-main thread while the asyncio event loop is running is pathologically slow (measured ~184 s vs ~20 s on the main thread), so the first `recall` can exceed the client's tool timeout. Use only if you cannot raise the startup timeout at all. |
| `off` | No preload; the model loads lazily on the first tool call (hits the same slow-thread-import path as `background`). |

If engram fails to connect at startup, raise the client's MCP startup timeout
(for Claude Code: `MCP_TIMEOUT=120000`) rather than switching to `background` —
that only converts a visible startup timeout into a 3-minute first `recall`.

---

## More like real memory (new in v0.3)

### Auto-encoding — sessions become memories by themselves

When a Claude Code session ends, a hook summarizes the conversation and saves
it as an episode memory (`engram setup` registers the hook for you). Even if
the agent forgets to call remember, "what we did yesterday" is preserved.
Disable with `auto_encode = false` in `config.toml`.

### Proactive recall — memory speaks up on its own

Every time you say something, a hook runs a lightweight search (a fast path
that never loads the embedding model) for related memories. The mode is set
by `surface_mode` in `config.toml`:

| Mode | Behavior |
|---|---|
| `shadow` (default) | Injects nothing; logs "this is what I would have surfaced" (for observation and tuning) |
| `active` | Actually injects strongly related memories into the agent's context |
| `off` | Does nothing |

The log lives at `~/.engram/surface/surface_log.jsonl`. We recommend watching
shadow mode for a while and switching to `active` once the surfaced candidates
look right. Use `engram surface "some text"` to check manually what would
surface.

Tuning parameters: `surface_threshold` (score threshold, default 0.45) /
`surface_min_relevance` (relevance floor, default 0.25 — a gate that keeps
even important memories from surfacing when they are unrelated to what you
said) / `surface_max_items` (max items per prompt, default 2).

### Memory rooms — separating work and personal contexts

Every memory carries a `room` label. Map folders to rooms in `config.toml`
and the room is resolved automatically from the working directory:

```toml
[room_paths]
'C:/Users/you/work-projects' = 'work'
'C:/Users/you/personal' = 'personal'
```

- Unmapped folders and pre-existing memories are all `common`
- recall searches only "current room + common" (`room="*"` searches across
  all rooms)
- Auto-encoding and proactive recall respect rooms too, so work memories
  never leak into personal contexts (and vice versa)

### Sharing memories across machines

The Markdown store can live on a synced folder (e.g. a cloud drive) shared by
several machines, but `index.db` is per-machine and local — so a memory written
on one machine isn't searchable on another until that machine indexes it. The
MCP server checks this at startup via `startup_index_check` in `config.toml`:
`auto` (default) reindexes when it detects a markdown/index mismatch, `warn`
logs a notice, `off` disables it. You can also run `engram reindex` any time.

---

## How the memory works

### Core design

Embeddings (where a memory sits in meaning-space) stay fixed; a separate
axis — **activation** — modulates search ranking. Meaning = where it is,
activation = how easily it comes to mind.

### The same properties as human memory

- **The more you use it, the easier it is to recall** — ACT-R activation
  model; every use by the agent reinforces it automatically
- **Unused memories sink but never disappear** — power-law decay; deep recall
  can always reach them through associative links
- **Memories from striking contexts are engraved deeply** — initial encoding
  boost by importance + slower decay = flashbulb memory
- **Corrected mistakes are engraved deepest of all** — the correct tool
  records the error together with the fix = hypercorrection effect

### Memory dynamics in brief

- Activation: `B = ln(Σ w_j·(now−t_j)^(−d_i))`, normalized to 0..1 with a
  sigmoid; computed on the fly from the access log
- `d_i = clamp(0.5 − 0.2·(imp−5)/5, 0.3, 0.6)` — higher importance forgets
  slower
- create event weight `1 + 2·(imp/10)` — critical memories start strong
- merely recalled: weight 0.3 / actually useful (reinforce): weight
  1.0×strength
- Memories reinforced together get co_recall links (Hebbian learning),
  growing an associative network that deep recall's spreading activation can
  traverse

### Search

Vector neighbors (Ruri-v3 embeddings) + BM25 full-text search merged with
RRF, then re-ranked by `0.6·relevance + 0.25·activation + 0.15·importance`.

#### Hybrid recall: exact tokens no longer sink

Candidate relevance now blends the two search paths instead of collapsing FTS
hits onto the vector similarity scale: vector hits keep their cosine
similarity, and FTS hits get a lexical relevance `1 - exp(bm25)` derived
directly from BM25 (0 for `bm25 >= 0`). When an id is hit by both, the higher
of the two wins. Rare, decisive lexical matches — memory IDs, file paths,
error codes, other exact tokens — now push `bm25` deep negative and surface
`lex` near 1.0, clearing the compressed 0.8–0.87 band where Ruri-v3 cosine
similarities tend to cluster. Previously, FTS-only hits were assigned the
minimum vector similarity among the candidate pool, which buried exact-match
results at the bottom of the ranking even when they were the obviously
correct answer.

### Memory types

| type | Contents |
|---|---|
| knowledge | Insights, solutions to problems, how to use tools |
| preference | The user's preferences, style, patterns in instructions |
| project | Goals, constraints, history, and background of the work |
| episode | A summary of what happened in a session |

### File layout

```
~/.engram/
  config.toml        Config file (generated by engram setup)
  index.db           SQLite index (rebuildable from Markdown via reindex)
  MEMORY_PROTOCOL.md Agent operating instructions (imported into each agent's instruction file)
  ONBOARDING.md      Initial interview script
  surface/           Proactive recall log and session state
  hooks.log          Hook activity log
  consolidation_state.json  Candidate-cluster count + last-nudge timestamp (consolidation nudge)
  perf/perf_log.jsonl       Timing log for MCP tool calls and startup (when perf_log = true)

<memories_dir>/      Source of truth: Markdown (opens and edits fine in Obsidian)
  knowledge/
  preferences/
  projects/
  episodes/YYYY/MM/
  _trash/
```

`memories_dir` defaults to `~/.engram/memories`, but pointing it at a Google
Drive or OneDrive synced folder gives you backup and multi-device sharing.
The SQLite index always stays local, so there is no sync-conflict risk.

### MCP tools

| Tool | When to use |
|---|---|
| `recall(query, mode, limit, type, room)` | At task start. fast = normal / deep = explores associative links, cold tier, and episodes / exhaustive = relevance-only full scan, ignoring activation, to dig up sunk memories |
| `remember(content, type, importance, tags, related_ids, room)` | When you learn an insight, preference, context, or event. importance 1–10 scores how critical the context is |
| `reinforce(ids, strength)` | At task end, report which memories actually helped (the nutrient for consolidation) |
| `correct(id, corrected_content, reason)` | When a memory was wrong. Use this, not forget (engraves the mistake itself deeply) |
| `link` / `forget` / `stats` / `reindex` | Auxiliary operations |
| `consolidation_candidates` / `mark_consolidated` | Consolidation (below) |

### Consolidation (the sleep of the system)

Clusters old episode memories and distills them into knowledge. The server
only returns candidates; the summarization is done by the LLM (your agent).
Example nightly run:

```powershell
claude -p "Call engram's consolidation_candidates, summarize each cluster with remember (type=knowledge or project, related_ids=the source episodes), finish with mark_consolidated, then report stats."
```

#### Automatic nudge cycle

On top of the cron-style nightly run above, engram nudges the agent to
consolidate on its own, without any scheduled job:

1. **SessionEnd** counts how many consolidation-candidate clusters currently
   exist (via `consolidation_candidates`) and stores the count in
   `data_dir/consolidation_state.json`.
2. **UserPromptSubmit** (the same lightweight hook that powers proactive
   recall) checks that state on the next session and, if enough clusters have
   piled up and enough time has passed since the last nudge, injects an
   `additionalContext` message asking the agent to run
   `consolidation_candidates` → `remember` → `mark_consolidated` at a natural
   pause in the conversation. The nudge fires even when `surface_mode = "off"`
   — it is independent of proactive recall.

Controlled by three settings (in `config.toml` or as `ENGRAM_*` environment
variables):

| Setting | Default | Meaning |
|---|---|---|
| `consolidate_nudge` | `true` | Master switch for the nudge cycle |
| `consolidate_nudge_min_clusters` | `3` | Minimum candidate clusters before nudging |
| `consolidate_nudge_interval_days` | `7.0` | Minimum time between nudges |

---

## For developers

Setting up to develop in this repository (run at the repo root):

```powershell
# Virtual env (for development; distribution uses uv tool install)
python -m venv "$env:USERPROFILE\.engram\venv"
& "$env:USERPROFILE\.engram\venv\Scripts\python.exe" -m pip install -e ".[dev]"
```

### Tests and verification

```powershell
$py = "$env:USERPROFILE\.engram\venv\Scripts\python.exe"

& $py -m pytest                          # all tests
& $py -m pytest tests\test_setup.py -q   # setup logic only
& $py scripts\simulate.py                # simulate access patterns (30 days)
& $py scripts\check_mcp_e2e.py           # MCP end-to-end check
```

### Diagnostics / CLI (for manual checks)

```powershell
$engram = "$env:USERPROFILE\.engram\venv\Scripts\engram.exe"

& $engram doctor
& $engram remember "content" --type knowledge --importance 7
& $engram recall "query" --deep
& $engram surface "utterance text"
& $engram stats
```

### Project structure

```
src/engram/
  config.py        Settings (defaults < config.toml < env vars) + room resolution
  engine.py        The memory engine
  store.py         Markdown source-of-truth store
  db.py            SQLite index (sqlite-vec + FTS5)
  dynamics.py      ACT-R activation model
  embedder.py      RuriEmbedder / FakeEmbedder
  server.py        MCP server (stdio)
  cli.py           CLI entry point
  setup.py         Setup wizard & doctor & hook registration
  hooks.py         Hook entry points (auto-encoding / proactive recall)
  transcript.py    Deterministic transcript summarization (auto-encoding)
  surface.py       Lightweight search path for proactive recall (no model)
  templates/       MEMORY_PROTOCOL.md / ONBOARDING.md
tests/
  test_setup.py    Setup pure-logic tests
  test_config.py   Settings precedence tests
  test_store.py    Markdown store tests
  test_db.py       DB operation tests
  test_engine.py   Engine tests
  test_room.py     Memory room tests
  test_surface.py  Proactive recall tests
  test_transcript.py  Transcript summarization tests
  test_hooks.py    Hook and hook-registration tests
  test_integration.py  Integration tests
```

---

## Uninstall

```powershell
# 1. Remove the registration from each agent
claude mcp remove engram

# 1b. Manually remove the engram entries from hooks in ~/.claude/settings.json
#     (the "engram hook ..." commands under SessionEnd / UserPromptSubmit)
# 2. Manually remove the engram block from ~/.claude/CLAUDE.md
# 3. Manually remove the [mcp_servers.engram] block from ~/.codex/config.toml
# 4. Manually remove the engram entry from ~/.gemini/config/mcp_config.json

# 5. Uninstall engram itself
uv tool uninstall engram

# 6. To delete the data as well (memories, config, model cache)
Remove-Item -Recurse -Force "$env:USERPROFILE\.engram"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--cl-nagoya--ruri*"
```

On macOS / Linux, steps 5–6 are:

```bash
uv tool uninstall engram
rm -rf ~/.engram
rm -rf ~/.cache/huggingface/hub/models--cl-nagoya--ruri*
```
