# Ollama — operational guide for the PoC

Minimal manual to manage the local models used as **targets** in the MIA-on-RAG
PoC. Ollama runs as a background daemon and exposes an HTTP API on
`http://localhost:11434` (the same one LangChain uses via `langchain-ollama`).

Model weights **do not** live in the Vault: they live in `~/.ollama/models`.
The Vault only holds the PoC code (`master/miarag/`) and this manual.

---

## 1. Prerequisites / installation

Installed via Homebrew (one-off):

```bash
brew install ollama
```

Version check:

```bash
ollama --version
```

---

## 2. Start / stop the daemon

Ollama must be running for `run`/`pull` commands and the API to respond.

```bash
# Foreground start (occupies the terminal, logs to screen):
ollama serve

# Or as a background service managed by Homebrew (auto-restarts on login):
brew services start ollama
brew services stop ollama      # stop the service
brew services list             # brew service status
```

Test that the daemon responds:

```bash
curl http://localhost:11434/api/version
```

---

## 3. Download a model (pull)

```bash
ollama pull llama3.1:8b        # primary PoC target (~4.9 GB)
ollama pull qwen2.5:3b         # small/fast model for iteration (~2 GB)
```

Practical notes:

- The pull downloads ~5 GB: **use stable WiFi**, not a hotspot. On unstable
  networks it may abort with `Error: max retries exceeded: unexpected EOF`.
- The pull is **resumable**: relaunching the same `pull` resumes from where it
  stopped, it does not restart from scratch.
- The tag after the colon is the default quantization/variant (for
  `llama3.1:8b` it's Q4_0). For different variants: see the model page on
  ollama.com/library.

---

## 4. Use a model

```bash
# Interactive chat (quit with /bye):
ollama run llama3.1:8b

# One-shot prompt:
ollama run llama3.1:8b "Summarize in one line what a RAG is."

# Via HTTP API (what LangChain does under the hood):
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello",
  "stream": false
}'
```

In the PoC code the model is selected via the `OLLAMA_TARGET_MODEL` env var in
`.env` (default `llama3.1:8b`), endpoint via `OLLAMA_BASE_URL`.

---

## 5. Inspect what is installed

```bash
ollama list                    # downloaded models + size
ollama ps                      # models currently loaded in RAM
ollama show llama3.1:8b        # details: params, quantization, template
du -sh ~/.ollama/models        # total disk usage
```

---

## 6. Cleanup / free space

Weights take several GB each. To reclaim space:

```bash
# Remove a single model (deletes its weights from disk):
ollama rm qwen2.5:3b

# Unload from RAM a model without deleting from disk:
ollama stop llama3.1:8b

# See what's taking space before deleting:
ollama list
du -sh ~/.ollama/models
```

Aggressive cleanup (careful: deletes EVERYTHING):

```bash
# Delete all downloaded models (you'll need to re-pull them):
rm -rf ~/.ollama/models/*

# Uninstall Ollama entirely (if no longer needed):
brew services stop ollama
brew uninstall ollama
rm -rf ~/.ollama              # remove residual models + config
```

Rule of thumb for the PoC: keep only the models you actively use as targets.
A `llama3.1:8b` (~5 GB) + one small model is enough; `ollama rm` the rest.

---

## 7. Common problems

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused` on :11434 | daemon not running | `ollama serve` or `brew services start ollama` |
| `max retries exceeded: unexpected EOF` during pull | unstable network (hotspot) | retry `ollama pull` on stable WiFi (resumes) |
| Extremely slow generation / swapping | model too large for 32 GB | use quantized 7–8B, not 70B |
| `model not found` | wrong tag or not pulled | `ollama list`, then `ollama pull <name>` |

Reference hardware: Apple M5, 32 GB unified RAM. Handles quantized 7–8B well;
70B models won't run locally — for those, use API targets (Bedrock/Azure) or EC2.
