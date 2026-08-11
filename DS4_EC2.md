# ds4 (DwarfStar) su EC2 — guida sperimentale

> **Fuori dallo scope tesi.** Questa guida NON riguarda il PoC MIA-su-RAG. È un
> appunto operativo per far girare un modello grande (DeepSeek V4 Flash / GLM 5.2)
> tramite l'engine **ds4 / DwarfStar** di antirez su una GPU cloud, se in futuro
> serve un target più capace del `llama3.1:8b` locale.
>
> ds4 **non** sostituisce Ollama per la PoC: è narrow (solo DeepSeek V4 + GLM 5.2,
> GGUF specifici), beta, non un GGUF runner generico. Vive fuori dal Vault, sulla
> EC2. Qui documento solo *come lanciarlo*, verificato dal README ufficiale
> (`github.com/antirez/ds4`, MIT, backend Metal/CUDA/ROCm).

---

## 0. Cos'è, in breve

- Inference engine in C, self-contained, ottimizzato per **DeepSeek V4 Flash** (e
  PRO su macchine ≥512GB), più **GLM 5.2**.
- Espone due binari: `ds4` (CLI + agent) e `ds4-server` (server HTTP con endpoint
  **OpenAI- e Anthropic-compatibili** → integrabile con LangChain via
  `langchain-openai` puntando all'endpoint locale, o via `/v1/messages`).
- Solo i GGUF ufficiali del repo `huggingface.co/antirez/deepseek-v4-gguf`
  funzionano: layout tensori/quantizzazione custom, non GGUF arbitrari.

---

## 1. Scelta istanza EC2

Riferimento richiesto: **g5.12xlarge**.

| Istanza | GPU | VRAM tot | vCPU | RAM | Arch CUDA | Note |
|---|---|---|---|---|---|---|
| g5.12xlarge | 4× NVIDIA A10G | 96 GB (4×24) | 48 | 192 GB | Ampere (sm_86) | economica; TP nativo non testato su Ampere → probabile fallback ordinato |
| g5.48xlarge | 8× A10G | 192 GB (8×24) | 192 | 768 GB | Ampere (sm_86) | come sopra, più VRAM/RAM (regge Flash q4) |
| **g6e.12xlarge** ⭐ | 4× L40S | 192 GB (4×48) | 48 | 384 GB | Ada (sm_89) | **consigliata**: L40S = arch testata dagli autori, TP nativo OK |
| g6e.48xlarge | 8× L40S | 384 GB (8×48) | 192 | 1536 GB | Ada (sm_89) | match esatto config di riferimento (8×L40S); cara |
| p5.48xlarge | 8× H100 | 640 GB | 192 | 2 TB | Hopper (sm_90) | overkill/costosissima per Flash; non necessaria |

**Consiglio in breve:**

- Strada **testata + veloce** → **g6e.12xlarge** (4× L40S, Ada `sm_89`). Match
  diretto con la config del progetto, `make cuda CUDA_ARCH=sm_89`, TP nativo attivo.
- **Spesa minima**, accetti di provare/adattare → g5.12xlarge (A10G Ampere): gira,
  ma rischio fallback ordinato (meno throughput). antirez suggerisce di usare un
  coding agent per adattare i kernel a hardware non-standard.
- **PRO** (512 GB) fuori portata di una singola istanza GPU → ignoralo, usa Flash q2/q4.

**Caveat hardware importante (verifica prima di impegnare $):**

- Il **tensor-parallel multi-GPU nativo** di ds4 è testato dagli autori su
  **8× L40S** (Ada Lovelace, `sm_89`) con ottimi risultati (120 t/s aggregati,
  2000 t/s prefill). L'A10G è **Ampere `sm_86`**, generazione precedente.
- Sull'A10G il codice può ricadere sul **"ordered exact fallback"** (concorrenza
  e correttezza garantite, ma **senza** lo speedup del batching nativo). Non è
  documentato come combinazione testata → aspettati di dover *provare* e magari
  chiedere a un coding agent di adattare i kernel (è esplicitamente il modello
  d'uso che antirez propone per hardware non-standard).
- Se il budget lo consente e vuoi la strada testata, **g6e.12xlarge (L40S / Ada)**
  è più vicina alla config di riferimento del progetto. `make cuda CUDA_ARCH=sm_89`
  nel README è proprio per L40S.

**AMI:** parti da una **Deep Learning AMI (Ubuntu)** con driver NVIDIA + CUDA
toolkit già installati — evita di installare i driver a mano. Verifica dopo il
boot con `nvidia-smi` (deve elencare tutte le GPU).

**Disco:** i GGUF Flash Q2/Q4 pesano ~40–160 GB. Alloca un volume **EBS gp3 da
almeno 300 GB** (o di più per PRO). Lo streaming SSD di ds4 vuole disco veloce.

---

## 2. Setup sulla EC2 (una volta)

```bash
# SSH nell'istanza (chiave e security group sono roba tua/personale):
ssh -i ~/percorso/chiave.pem ubuntu@<IP-PUBBLICO-EC2>

# Toolchain di build (se non già presente nell'AMI):
sudo apt-get update && sudo apt-get install -y build-essential git

# Verifica GPU e driver:
nvidia-smi

# Clona ds4:
git clone https://github.com/antirez/ds4.git
cd ds4
```

Build CUDA (scegli in base alla GPU):

```bash
make cuda-generic          # GPU CUDA generiche (A10G incluse)
# oppure, per L40S / Ada, come nel README:
make cuda CUDA_ARCH=sm_89
# make cuda-spark          # solo DGX Spark / GB10
```

---

## 3. Scaricare un modello

```bash
# DeepSeek V4 Flash — scegli UNA quantizzazione:
./download_model.sh ds4f-q2      # ~macchine 96/128 GB
./download_model.sh ds4f-q4      # >= 256 GB RAM (più qualità, più pesante)

# GLM 5.2 (alternativa):
./download_model.sh glm-antirez-q4
```

Lo script scarica in `./gguf/`, riprende download parziali (`curl -C -`), e
aggiorna il symlink `./ds4flash.gguf` al modello scelto. Token HF opzionale
(`HF_TOKEN` o `--token`) per download pubblici.

**Nota RAM/VRAM:** su g5.12xlarge (96 GB VRAM) il Flash q2 può stare quasi
residente; per q4 o context lunghi valuta lo **SSD streaming** (§5) o passa a
g5.48xlarge / g6e con più VRAM.

---

## 4. Avviare il server (per usarlo da LangChain)

`ds4-server` espone endpoint OpenAI- e Anthropic-compatibili.

Single-GPU / test rapido:

```bash
./ds4-server --ctx 100000 --host 0.0.0.0
```

Multi-GPU (4× A10G su g5.12xlarge) — tenta il tensor-parallel:

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

- `--host 0.0.0.0` espone sulla rete → **apri la porta SOLO verso il tuo IP** nel
  security group, mai `0.0.0.0/0`. Meglio ancora: tienilo su `127.0.0.1` e usa un
  tunnel SSH (§6).
- `--batched-session N`: N sessioni KV residenti concorrenti. Scegli N e `--ctx`
  così che stiano tutte in VRAM.
- `--cors` solo se un browser da altra origine deve chiamarlo.

Endpoint disponibili: `/v1/models`, `/v1/chat/completions`, `/v1/responses`,
`/v1/completions`, `/v1/messages` (Anthropic-style, per client Claude-Code-like).

`./ds4-server --help` per la lista completa dei flag.

### Integrazione LangChain (dal tuo Mac, via tunnel)

Puntando `langchain-openai` all'endpoint OpenAI-compatibile:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",   # tunnel SSH verso la EC2 (vedi §6)
    api_key="not-needed",                   # ds4-server non richiede chiave
    model="deepseek-v4-flash",
)
```

(Adatta porta/model id a quanto stampa `ds4-server` all'avvio.)

---

## 5. Modello più grande della RAM — SSD streaming

Se il GGUF non entra in VRAM/RAM, ds4 tiene residenti i pesi non-routed e
streamma gli expert MoE da disco:

```bash
./ds4 -m ./ds4flash.gguf --ssd-streaming
# budget cache expert esplicito se l'auto è troppo grande:
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

Più lento del full-resident, ma utile per far girare Flash q4 / PRO su VRAM
limitata. Il prefill resta veloce; la generazione soffre di più i cache miss.

---

## 6. Accesso sicuro (tunnel SSH invece di esporre la porta)

Preferibile a `--host 0.0.0.0` + porta aperta. Sul Mac:

```bash
# Tieni il server su 127.0.0.1 nella EC2, poi:
ssh -i ~/chiave.pem -N -L 8080:localhost:8080 ubuntu@<IP-EC2>
# Ora http://localhost:8080 sul Mac punta al ds4-server sulla EC2.
```

Così l'API non è mai esposta su Internet.

---

## 7. Costi / spegnere l'istanza (IMPORTANTE)

Le g5/g6e costano **diversi $/ora**. Regole:

```bash
# Fermare l'istanza quando non la usi (NON la cancella, EBS resta a pagamento minimo):
aws ec2 stop-instances --instance-ids <ID>

# Riavviare:
aws ec2 start-instances --instance-ids <ID>

# Terminare del tutto (cancella l'istanza; l'EBS root va via se DeleteOnTermination=true):
aws ec2 terminate-instances --instance-ids <ID>
```

- **Ferma sempre l'istanza a fine sessione** — una g5.12xlarge dimenticata accesa
  brucia budget di notte.
- I GGUF sull'EBS restano a pagamento anche a istanza ferma: se non ti servono più,
  cancella il volume o fai uno snapshot e termina.
- Infrastruttura, chiavi, security group, billing = **roba personale tua**, fuori
  dal repo e fuori dal Vault.

---

## 8. Perché NON è la soluzione per la PoC

- ds4 è specializzato su DeepSeek V4 / GLM 5.2: non ti dà `llama3.1` né controllo
  fine su log-probs/perplexity nel modo che serve a S2MIA/BudgetLeak.
- Beta, in rapido cambiamento, richiede build da sorgente e tuning hardware.
- Per il PoC MIA-su-RAG resta **Ollama locale** (target primario) + **API
  Bedrock/Azure** (target black-box realistici). Vedi `OLLAMA.md` e CLAUDE.md.

Questa guida è solo un "se un giorno ti serve un modello grosso su GPU cloud".
