# Ollama — guida operativa per la PoC

Manuale minimale per gestire i modelli locali usati come **target** della PoC MIA-su-RAG.
Ollama gira come daemon in background ed espone un'API HTTP su `http://localhost:11434`
(la stessa che LangChain usa via `langchain-ollama`).

I pesi dei modelli **non** stanno nel Vault: vivono in `~/.ollama/models`. Il Vault contiene
solo il codice PoC (`master/poc/`) e questo manuale.

---

## 1. Prerequisiti / installazione

Installato via Homebrew (una volta sola):

```bash
brew install ollama
```

Verifica versione:

```bash
ollama --version
```

---

## 2. Avviare / fermare il daemon

Ollama deve essere in esecuzione perché i comandi `run`/`pull` e l'API rispondano.

```bash
# Avvio in foreground (occupa il terminale, log a schermo):
ollama serve

# Oppure come servizio in background gestito da Homebrew (si riavvia da solo al login):
brew services start ollama
brew services stop ollama      # ferma il servizio
brew services list             # stato dei servizi brew
```

Test che il daemon risponda:

```bash
curl http://localhost:11434/api/version
```

---

## 3. Scaricare un modello (pull)

```bash
ollama pull llama3.1:8b        # target primario PoC (~4.9 GB)
ollama pull qwen2.5:3b         # modello piccolo/veloce per iterare (~2 GB)
```

Note pratiche:

- Il pull scarica ~5 GB: **fallo su WiFi stabile**, non su hotspot. Su rete instabile
  può interrompersi con `Error: max retries exceeded: unexpected EOF`.
- Il pull è **ripristinabile**: rilanciando lo stesso `pull` riprende da dove si era
  fermato, non riparte da zero.
- Il tag dopo i due punti è la quantizzazione/variante di default (per `llama3.1:8b`
  è Q4_0). Per varianti diverse: vedi la pagina del modello su ollama.com/library.

---

## 4. Usare un modello

```bash
# Chat interattiva (esci con /bye):
ollama run llama3.1:8b

# Prompt secco one-shot:
ollama run llama3.1:8b "Riassumi in una riga cos'è un RAG."

# Via API HTTP (quello che fa LangChain sotto il cofano):
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Ciao",
  "stream": false
}'
```

Nel codice PoC il modello si seleziona con la variabile `OLLAMA_TARGET_MODEL` in `.env`
(default `llama3.1:8b`), endpoint da `OLLAMA_BASE_URL`.

---

## 5. Ispezionare cosa c'è installato

```bash
ollama list                    # modelli scaricati + dimensione
ollama ps                      # modelli attualmente caricati in RAM
ollama show llama3.1:8b        # dettagli: parametri, quantizzazione, template
du -sh ~/.ollama/models        # spazio totale occupato su disco
```

---

## 6. Pulizia / liberare spazio (clean)

I pesi occupano diversi GB l'uno. Per recuperare spazio:

```bash
# Rimuovere un singolo modello (cancella i suoi pesi da disco):
ollama rm qwen2.5:3b

# Scaricare dalla RAM un modello caricato senza cancellarlo da disco:
ollama stop llama3.1:8b

# Vedere cosa occupa spazio prima di cancellare:
ollama list
du -sh ~/.ollama/models
```

Pulizia più aggressiva (attenzione: cancella TUTTO):

```bash
# Cancella tutti i modelli scaricati (dovrai ri-pullarli):
rm -rf ~/.ollama/models/*

# Disinstallare Ollama del tutto (se non serve più):
brew services stop ollama
brew uninstall ollama
rm -rf ~/.ollama              # rimuove modelli + config residui
```

Regola pratica per la PoC: tieni solo i modelli che stai usando come target.
Un `llama3.1:8b` (~5 GB) + un modello piccolo bastano; il resto `ollama rm`.

---

## 7. Problemi comuni

| Sintomo | Causa probabile | Rimedio |
|---|---|---|
| `connection refused` su :11434 | daemon non attivo | `ollama serve` o `brew services start ollama` |
| `max retries exceeded: unexpected EOF` durante pull | rete instabile (hotspot) | rilancia `ollama pull` su WiFi stabile (riprende) |
| Generazione lentissima / swap | modello troppo grande per 32 GB | usa 7-8B quantizzato, non 70B |
| `model not found` | tag sbagliato o non pullato | `ollama list`, poi `ollama pull <nome>` |

Hardware di riferimento: Apple M5, 32 GB unified RAM. Regge bene 7-8B quantizzati;
i 70B non girano in locale — per quelli usa i target via API (Bedrock/Azure) o EC2.
