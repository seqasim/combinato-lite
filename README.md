# Combinato-Lite

Lightweight, portable rewrite of [Combinato](https://github.com/jniediek/combinato) spike sorting.

Same core algorithms (wavelet features → SPC clustering → template matching → artifact rejection → grouping). Modern packaging, h5py instead of PyTables, a single CLI, and a browser-based curation GUI.

Works on **macOS** (Apple Silicon via Rosetta 2 for SPC), **Windows**, and **Linux**.

## Install

```bash
git clone <this-repo> combinato-lite
cd combinato-lite
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

- **CLI command:** `combinato`
- **Python import:** `import combinato_lite` (never collides with classic Combinato)

Pinned dependency versions used in CI/dev: [`requirements.lock.txt`](requirements.lock.txt).

### Apple Silicon note

The bundled SPC clustering binary is x86_64. On Apple Silicon it runs under **Rosetta 2**:

```bash
softwareupdate --install-rosetta
```

If Rosetta is missing, `combinato cluster` prints a clear error.

## Quickstart

```bash
# 1) Extract (parallel over time chunks; default -j 4)
combinato extract --files CSC01.ncs CSC02.ncs -j 8
# combinato extract --matfile recording.mat
# combinato extract --h5 --files continuous.h5 -j 8

# 2) One-shot sort (clusters sessions in parallel when a file is split)
combinato sort --datafile CSC01/data_CSC01.h5 --label myrun -j 8

# Or step by step:
combinato prepare --datafile CSC01/data_CSC01.h5 --label myrun
combinato cluster --jobs sort_pos_myrun.txt -j 8
combinato combine --datafile CSC01/data_CSC01.h5 \
  --sessions sort_pos_myrun_0000000_0004999 \
  --label sort_pos_myrun

# 3) Manual curation GUI
combinato gui --datafile CSC01/data_CSC01.h5 --sorting CSC01/sort_pos_myrun
```

Open the printed URL (default `http://127.0.0.1:8765`).

## Parallelization

Combinato-Lite parallelizes the two expensive stages. Use **`-j` / `--workers`** everywhere it matters.

| Stage | What is parallel? | How to control |
|-------|-------------------|----------------|
| **extract** | Time blocks (and multi-file job lists) across processes | `combinato extract … -j N` (default **4**) |
| **prepare** | Serial (fast). Splits long spike trains into sessions of `--max-nspk` spikes | `--max-nspk` (default ~20000) |
| **cluster** | **One process per session** (each runs SPC) | `combinato cluster … -j N` (default: all CPUs, capped by #jobs). `--single` or `-j 1` = serial |
| **sort** | Same as cluster for the clustering stage | `combinato sort … -j N` |
| **combine / gui** | Single-process | — |

**Rule of thumb on a workstation or interactive node**

```bash
# Match -j to the CPUs you actually have free
combinato extract --files *.ncs -j 16
combinato prepare --datafile CSC01/data_CSC01.h5 --label run1
combinato cluster --jobs sort_pos_run1.txt -j 16
```

**Many channels on a cluster (AMAREL / SLURM)** — two patterns:

1. **One fat job** (many workers inside one allocation):

```bash
#SBATCH -N 1 -c 24 -t 4:00:00
combinato extract --files $(cat channels.txt) -j $SLURM_CPUS_PER_TASK
combinato sort --datafile CSC01/data_CSC01.h5 --label amarel -j $SLURM_CPUS_PER_TASK
```

2. **Job array** (one channel or one session per SLURM task; use `-j 1` inside each task):

```bash
#SBATCH --array=1-100 -c 1
CHAN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" channels.txt)
combinato extract --files "$CHAN" -j 1
# after all extracts finish:
# combinato prepare … && combinato cluster --jobs … -j 1   # or -j >1 if several sessions per channel
```

`prepare` is what creates the parallelizable units for clustering: more spikes → more sessions → more jobs that `-j` can spread across cores.

## Remote / AMAREL (Rutgers)

Install in your conda/venv on the cluster the same way (`pip install -e .`).

```bash
combinato sort --datafile /scratch/$USER/CSC01/data_CSC01.h5 --label amarel -j $SLURM_CPUS_PER_TASK
```

GUI over SSH (port-forward from your laptop):

```bash
# on the cluster
combinato gui --datafile ... --sorting ... --host 127.0.0.1 --port 8765

# on your laptop
ssh -L 8765:localhost:8765 <netid>@amarel.rutgers.edu
```

Then open `http://127.0.0.1:8765` locally.

## Configuration

Defaults live in `combinato_lite.config.CombinatoConfig`. Override with:

1. `$COMBINATO_CONFIG`, or `./combinato_lite.yaml`, or `./combinato.yaml`
2. Environment variables `COMBINATO_<FieldName>` (e.g. `COMBINATO_nFeatures=12`)
3. CLI `--config path.yaml` / `--verbose`

See [`combinato.yaml.example`](combinato.yaml.example).

## CLI reference

| Command | Purpose | Parallel? |
|---------|---------|-----------|
| `combinato extract` | Spike detection from `.ncs` / `.mat` / HDF5 | Yes (`-j`) |
| `combinato prepare` | Build session folders + job list | No (creates jobs for cluster) |
| `combinato cluster` | SPC + template match per session | Yes (`-j`) |
| `combinato combine` | Concatenate sessions + group | No |
| `combinato sort` | prepare → cluster → combine | Cluster stage yes (`-j`) |
| `combinato gui` | Local web curation UI | No |

## HDF5 layout (compatible with classic Combinato)

**`data_<name>.h5`**

- `/pos/spikes` `(N, 64) float32`, `/pos/times` `(N,) float64`
- `/neg/spikes`, `/neg/times`
- `/thr` `(M, 3)` extraction thresholds

**`sort_*/sort_cat.h5`**

- `index`, `classes`, `matches`, `artifacts`, `groups` `(clid, gid)`, `types` `(gid, type)`
- root attribute `sign` = `pos` \| `neg`

Existing Combinato data files are readable (standard HDF5; h5py replaces PyTables).

## GUI features

- Browse groups with mean / sample waveforms (Plotly)
- Spike-time histogram
- Auto / cross-correlogram (pure NumPy; no Cython `.so`)
- Merge groups
- Assign SU / MU / Artifact / Unassigned (saved immediately to `sort_cat.h5`)

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

```python
import combinato_lite
from combinato_lite.config import CombinatoConfig
```

CI runs on Ubuntu, macOS, and Windows (Python 3.10 / 3.12). SPC-dependent tests skip cleanly if the binary cannot run.

## Relationship to upstream Combinato

Algorithms are ported from Johannes Niediek's Combinato (MIT). Cite the original paper when publishing:

> Niediek et al., PLOS ONE 2016. doi:10.1371/journal.pone.0166598

Classic Combinato can remain installed side-by-side; this project imports as `combinato_lite` only.

## License

MIT — see [LICENSE](LICENSE).
