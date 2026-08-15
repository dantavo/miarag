# ds4 (DwarfStar) on EC2 — experimental guide

> **Out of thesis scope.** This guide is NOT about the MIA-on-RAG PoC. It's an
> operational note for running a large model (DeepSeek V4 Flash / GLM 5.2) via
> antirez's **ds4 / DwarfStar** engine on a cloud GPU, in case a more capable
> target than local `llama3.1:8b` is needed in the future.
>
> ds4 **does not** replace Ollama for the PoC: it is narrow (only DeepSeek V4 +
> GLM 5.2, specific GGUFs), beta, not a generic GGUF runner. It lives outside
> the Vault, on the EC2. Here I only document *how to launch it*, verified
> against the official README (`github.com/antirez/ds4`, MIT, Metal/CUDA/ROCm
> backends).

---

## 0. What it is, briefly

- Inference engine in C, self-contained, optimized for **DeepSeek V4 Flash**
  (and PRO on ≥512 GB machines), plus **GLM 5.2**.
- Ships two binaries: `ds4` (CLI + agent) and `ds4-server` (HTTP server with
  **OpenAI- and Anthropic-compatible** endpoints → integrates with LangChain
  via `langchain-openai` pointing at the local endpoint, or via `/v1/messages`).
- Only the official GGUFs from `huggingface.co/antirez/deepseek-v4-gguf` work:
  custom tensor layout/quantization, not arbitrary GGUFs.

---

## 1. EC2 instance choice

Requested reference: **g5.12xlarge**.

| Instance | GPU | Total VRAM | vCPU | RAM | CUDA arch | Notes |
|---|---|---|---|---|---|---|
| g5.12xlarge | 4× NVIDIA A10G | 96 GB (4×24) | 48 | 192 GB | Ampere (sm_86) | cheap; native TP untested on Ampere → likely ordered fallback |
| g5.48xlarge | 8× A10G | 192 GB (8×24) | 192 | 768 GB | Ampere (sm_86) | as above, more VRAM/RAM (fits Flash q4) |
| **g6e.12xlarge** ⭐ | 4× L40S | 192 GB (4×48) | 48 | 384 GB | Ada (sm_89) | **recommended**: L40S = arch tested by the authors, native TP OK |
| g6e.48xlarge | 8× L40S | 384 GB (8×48) | 192 | 1536 GB | Ada (sm_89) | exact match for reference config (8×L40S); expensive |
| p5.48xlarge | 8× H100 | 640 GB | 192 | 2 TB | Hopper (sm_90) | overkill/very expensive for Flash; not needed |

**Short recommendation:**

- **Tested + fast** path → **g6e.12xlarge** (4× L40S, Ada `sm_89`). Direct
  match with the project's config, `make cuda CUDA_ARCH=sm_89`, native TP
  active.
- **Minimum spend**, willing to experiment/adapt → g5.12xlarge (A10G Ampere):
  runs, but risks ordered fallback (lower throughput). antirez suggests using
  a coding agent to adapt kernels to non-standard hardware.
- **PRO** (512 GB) out of reach for a single GPU instance → ignore it, use
  Flash q2/q4.

**Important hardware caveats (verify before committing $):**

- Native **multi-GPU tensor-parallel** in ds4 is tested by the authors on
  **8× L40S** (Ada Lovelace, `sm_89`) with excellent results (120 t/s
  aggregate, 2000 t/s prefill). A10G is **Ampere `sm_86`**, previous gen.
- On A10G the code may fall back to the **"ordered exact fallback"**
  (concurrency and correctness guaranteed, but **without** the native
  batching speedup). Not documented as a tested combination → expect to
  *try* and possibly ask a coding agent to adapt the kernels (this is
  explicitly the usage model antirez proposes for non-standard hardware).
- If budget allows and you want the tested path, **g6e.12xlarge (L40S /
  Ada)** is closer to the reference project config. `make cuda
  CUDA_ARCH=sm_89` in the README is precisely for L40S.

**AMI:** start from a **Deep Learning AMI (Ubuntu)** with NVIDIA drivers +
CUDA toolkit pre-installed — avoid installing drivers by hand. Verify after
boot with `nvidia-smi` (must list all GPUs).

**Disk:** Flash Q2/Q4 GGUFs weigh ~40–160 GB. Allocate an **EBS gp3 volume of
at least 300 GB** (more for PRO). ds4's SSD streaming wants fast disk.

---

## 2. EC2 setup (one-off)

```bash
# SSH into the instance (key and security group are your personal stuff):
ssh -i ~/path/to/key.pem ubuntu@<EC2-PUBLIC-IP>

# Build toolchain (if not already in the AMI):
sudo apt-get update && sudo apt-get install -y build-essential git

# GPU + driver check:
nvidia-smi

# Clone ds4:
git clone https://github.com/antirez/ds4.git
cd ds4
```

CUDA build (choose based on GPU):

```bash
make cuda-generic          # generic CUDA GPUs (A10G included)
# or, for L40S / Ada, as in the README:
make cuda CUDA_ARCH=sm_89
# make cuda-spark          # only DGX Spark / GB10
```

---

## 3. Download a model

```bash
# DeepSeek V4 Flash — pick ONE quantization:
./download_model.sh ds4f-q2      # ~machines with 96/128 GB
./download_model.sh ds4f-q4      # >= 256 GB RAM (better quality, heavier)

# GLM 5.2 (alternative):
./download_model.sh glm-antirez-q4
```

The script downloads to `./gguf/`, resumes partial downloads (`curl -C -`),
and updates the `./ds4flash.gguf` symlink to the chosen model. Optional HF
token (`HF_TOKEN` or `--token`) for public downloads.

**RAM/VRAM note:** on g5.12xlarge (96 GB VRAM) Flash q2 can be nearly
resident; for q4 or long contexts consider **SSD streaming** (§5) or move to
g5.48xlarge / g6e with more VRAM.

---

## 4. Start the server (for LangChain integration)

`ds4-server` exposes OpenAI- and Anthropic-compatible endpoints.

Single-GPU / quick test:

```bash
./ds4-server --ctx 100000 --host 0.0.0.0
```

Multi-GPU (4× A10G on g5.12xlarge) — attempt native tensor-parallel:

```bash
MODEL=./ds4flash.gguf

./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,1,2,3 \
  --model "$MODEL" \
  --ctx 100000 \
  --batched-session 8 \
  --host 0.0.0.0
```

- `--host 0.0.0.0` exposes on the network → **open the port ONLY to your IP**
  in the security group, never `0.0.0.0/0`. Better still: keep it on
  `127.0.0.1` and use an SSH tunnel (§6).
- `--batched-session N`: N concurrent resident KV sessions. Choose N and
  `--ctx` so they all fit in VRAM.
- `--cors` only if a browser from a different origin has to call it.

Available endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/responses`,
`/v1/completions`, `/v1/messages` (Anthropic-style, for Claude-Code-like
clients).

`./ds4-server --help` for the full flag list.

### LangChain integration (from your Mac, via tunnel)

Point `langchain-openai` at the OpenAI-compatible endpoint:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",   # SSH tunnel to EC2 (see §6)
    api_key="not-needed",                   # ds4-server needs no key
    model="deepseek-v4-flash",
)
```

(Adapt port/model id to whatever `ds4-server` prints at startup.)

---

## 5. Model larger than RAM — SSD streaming

If the GGUF doesn't fit VRAM/RAM, ds4 keeps non-routed weights resident and
streams MoE experts from disk:

```bash
./ds4 -m ./ds4flash.gguf --ssd-streaming
# explicit expert cache budget if auto is too large:
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

Slower than full-resident but useful to run Flash q4 / PRO on limited VRAM.
Prefill stays fast; generation suffers more from cache misses.

---

## 6. Secure access (SSH tunnel instead of exposing the port)

Preferable to `--host 0.0.0.0` + open port. On the Mac:

```bash
# Keep the server on 127.0.0.1 on the EC2, then:
ssh -i ~/key.pem -N -L 8080:localhost:8080 ubuntu@<EC2-IP>
# Now http://localhost:8080 on the Mac points at ds4-server on the EC2.
```

This way the API is never exposed to the internet.

---

## 7. Costs / stopping the instance (IMPORTANT)

g5/g6e instances cost **several $/hour**. Rules:

```bash
# Stop the instance when not in use (does NOT delete it, EBS keeps its minimal cost):
aws ec2 stop-instances --instance-ids <ID>

# Restart:
aws ec2 start-instances --instance-ids <ID>

# Terminate entirely (deletes the instance; root EBS goes away if DeleteOnTermination=true):
aws ec2 terminate-instances --instance-ids <ID>
```

- **Always stop the instance at end of session** — a forgotten g5.12xlarge
  burns budget overnight.
- GGUFs on EBS keep costing even when the instance is stopped: if you no
  longer need them, delete the volume or take a snapshot and terminate.
- Infrastructure, keys, security groups, billing = **personal stuff of
  yours**, outside the repo and outside the Vault.

---

## 8. Why this is NOT the solution for the PoC

- ds4 is specialized on DeepSeek V4 / GLM 5.2: it does not give you
  `llama3.1` nor the fine control over log-probs / perplexity needed by
  S2MIA/BudgetLeak.
- Beta, rapidly changing, requires build from source and hardware tuning.
- For the MIA-on-RAG PoC the target remains **local Ollama** (primary target)
  + **Bedrock/Azure APIs** (realistic black-box targets). See `OLLAMA.md`
  and CLAUDE.md.

This guide is only a "if one day you need a large model on cloud GPU".
