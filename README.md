# apple-sme-gemm

Confronto minimo e riproducibile fra **SME ottimizzato**, **SME32 originale**
e **Apple Accelerate**. C con intrinsics, matrici quadrate FP32, un thread.
I kernel sono quelli verificati sul MacBook Pro M4 Pro; il progetto è autonomo.

![Throughput e speedup su M4 Pro](examples/m4-pro/plots/comparison.png)

## Avvio

Richiede macOS 15+, Apple Silicon con SME2 e vettori streaming da 512 bit,
Command Line Tools con supporto a `apple-m4` (verificato con Apple Clang 17)
e Python 3.10+. Build, test e benchmark usano soltanto strumenti di sistema
e la libreria standard Python; solo i grafici richiedono Matplotlib.

```sh
python3 build.py
python3 check.py
python3 benchmark.py --suite full --output results/run1
python3 plot.py results/run1
```

Per installare la dipendenza dei grafici in un ambiente locale:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python plot.py results/run1
```

Le cartelle di output devono essere nuove. Per un giro breve, sostituisci
il comando di benchmark con:

```sh
python3 benchmark.py --suite quick --rounds 2 --samples 5 --min-sample-ms 20 --output results/quick
```

Sono disponibili anche `make build`, `make test`, `make quick`, `make bench`
e `make plot`. `make bench plot OUT=results/run2` compila, misura e genera
i grafici; puoi scegliere l'interprete con `PYTHON=.venv/bin/python`.

## Implementazioni

| Backend | Strategia |
| --- | --- |
| `sme32` | Originale 32×32, quattro tile ZA, packing A16/B16, K-blocking 256 |
| `sme_opt` | Fino a N=512: 32×32, trasposizione di A in ZA e B row-major senza packing. Oltre: 16×64, caricamenti SME2 a quattro vettori, packing A16/B64 e blocking K=2048, M=256, N=512 |
| `accelerate` | `cblas_sgemm`, BLAS esplicitamente impostata a un thread |

Per N>512 non divisibile per 64, il backend ottimizzato usa il percorso
SME32 originale. Non ci sono padding o deleghe ad Accelerate nei kernel SME.
Le dimensioni ammesse sono multipli di 32 da 32 a 8192. I campi
`kernel_variant` e `packing_layout` nel JSON mostrano il percorso selezionato.
Il packing di B grande usa blocchi da 256 colonne e 32 righe. L'unrolling del
kernel è di quattro iterazioni, senza il precedente software pipelining che
generava copie di registri extra con Clang 17.

## Cosa misurano i grafici

| Modalità | Timer |
| --- | --- |
| `total` | Packing richiesto dal backend + prodotto completo; Accelerate riceve gli input row-major |
| `compute` | Solo prodotto con input già preparati; nel kernel ottimizzato piccolo B resta row-major |

Accelerate compare solo in `total`, perché non espone l'interfaccia di packing
usata dai nostri kernel. Il confronto principale è **packing incluso**.
Allocazione, inizializzazione, avvio dei processi, calibrazione e verifica
numerica sono fuori dal timer. I buffer sono allineati a 64 byte.

- **Throughput**: `2*N³ / tempo_in_secondi / 10⁹`, in GFLOP/s.
- **Speedup**: `tempo_baseline / tempo_backend`. Sopra 1 il backend è più veloce.
  Il riferimento è Accelerate in `total`, SME32 in `compute`.
- Lo speedup è calcolato appaiando le mediane dei backend nello stesso round;
  si riporta la mediana dei rapporti. La fascia è l'IQR fra round, non un
  intervallo di confidenza. Il rapporto fra le mediane globali può differire.

La suite completa usa N=32…4096, tre round e sette campioni per job. Casi e
backend vengono ordinati casualmente con seed riproducibile. Ogni campione
è una media di batch calibrati a circa 30 ms o più; non misura la latenza
di coda delle singole chiamate. Si riutilizzano gli input: è un test a regime.
QoS `USER_INITIATED` favorisce i core veloci, ma lo scheduler macOS decide
il core; mantieni alimentazione, modalità energetica e carico concorrente coerenti.

Il confronto include differenze di packing e algoritmo: misura le implementazioni,
non il solo effetto isolato dell'ISA. Non descrive il throughput multicore massimo.

## Risultati e controlli

`plots/comparison.png`, `.svg` e `.pdf` contengono throughput e speedup per
entrambe le modalità. La cartella della run contiene inoltre:

| File | Contenuto |
| --- | --- |
| `summary.csv` | Tempi mediani, quartili e throughput per backend |
| `speedups.csv` | Speedup appaiati contro SME32 e Accelerate |
| `raw.jsonl`, `samples.csv` | Tutti i risultati grezzi e i campioni |
| `verification.csv` | Controlli numerici e fingerprint degli input |
| `environment.json` | Hardware, compilatore, opzioni e hash dei sorgenti/eseguibile |
| `status.json` | Completezza della suite |

`Ctrl+C` preserva i job completati e i report parziali; `failed_job.json`
registra il comando riproducibile se un job fallisce.

`check.py` compila ed esegue test con AddressSanitizer e UndefinedBehaviorSanitizer:
packing completo su 12 dimensioni; prodotti su 15 dimensioni, due semi e due
sovrascritture, con guardie sull'output e riferimento scalare double.
Controlla ogni elemento fino a N=192 e 512 elementi nei casi grandi.
I test Python verificano pairing, metriche, report senza timing e dry-run.
Ogni job del benchmark verifica inoltre l'output prima e dopo i campioni
(tutti gli elementi fino a N=128, altrimenti 256) e tutti i valori finiti.

Il build verifica nel disassemblato i quattro tile ZA e i caricamenti
multi-vector del kernel ottimizzato. Il runner rifiuta eseguibili o sorgenti
modificati rispetto al manifest: dopo modifiche al C, esegui `python3 build.py`.

Per selezioni personalizzate o sola verifica numerica:

```sh
python3 benchmark.py --sizes 32,64,96,320,544,1088,2080,2112 --verify-only --output results/checks
python3 benchmark.py --sizes 512,1024 --modes total --output results/custom
python3 benchmark.py --suite full --dry-run
```

## Esempio verificato

[`examples/m4-pro/`](examples/m4-pro/) contiene una nuova esecuzione completa
dalla repository stessa: dati grezzi, metadati e grafici. Non è un risultato
universale: differenze piccole vanno valutate insieme alla dispersione.
La variante ottimizzata non supera Accelerate su ogni dimensione.


Misura locale su M4 Pro, 48 GB, Apple Clang 17; packing incluso:

| N | SME32 GFLOP/s | Ottimizzato GFLOP/s | Accelerate GFLOP/s | Speedup ottimizzato / Accelerate |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 640 | 1539 | 1407 | 1.09× |
| 512 | 1342 | 1797 | 1792 | 1.00× |
| 1024 | 1296 | 1702 | 1668 | 1.02× |
| 2048 | 1095 | 1602 | 1696 | 0.95× |
| 4096 | 945 | 1576 | 1571 | 1.01× |

Sono 120 job e 840 campioni, tutti verificati. A N=2048 il kernel ottimizzato
resta circa 6% più lento di Accelerate in questa esecuzione; a N=4096 i risultati sono sostanzialmente
alla pari. Le differenze di pochi punti percentuali non implicano un vantaggio
significativo senza considerare la variabilità.

I kernel sono in `src/sme_opt.c`, `src/sme32.c`; il packing in `src/pack*.c`
e `src/sme_pack.c`. `main.c` gestisce timing e verifica; i quattro script
Python gestiscono compilazione, test, benchmark e grafici. Non è richiesta
alcuna dipendenza da Eigen o altre librerie BLAS esterne.
