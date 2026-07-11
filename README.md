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

One command. No `setup_options.py`, no generated `options.py`, no `sys.path` hacks.

> **Note:** The import package is still named `combinato`. If you also have classic Combinato on `PYTHONPATH` or installed in the same environment, uninstall or remove it first so this package is the one that loads.

Pinned dependency versions used in CI/dev are recorded in [`requirements.lock.txt`](requirements.lock.txt) (`pip install -r requirements.lock.txt && pip install -e .`).

### Apple Silicon note

The bundled SPC clustering binary is x86_64. On Apple Silicon it runs under **Rosetta 2**:

```bash
softwareupdate --install-rosetta
```

If Rosetta is missing, `combinato cluster` prints a clear error.

## Quickstart

```bash
# 1) Extract spikes from Neuralynx / MATLAB / continuous HDF5
combinato extract --files CSC01.ncs CSC02.ncs
# combinato extract --matfile recording.mat
# combinato extract --h5 --files continuous.h5

# 2) One-shot sort (prepare + cluster + combine + group)
combinato sort --datafile CSC01/data_CSC01.h5 --label myrun

# Or step by step:
combinato prepare --datafile CSC01/data_CSC01.h5 --label myrun
combinato cluster --jobs sort_pos_myrun.txt
combinato combine --datafile CSC01/data_CSC01.h5 \
  --sessions sort_pos_myrun_0000000_0004999 \
  --label sort_pos_myrun

# 3) Manual curation GUI
combinato gui --datafile CSC01/data_CSC01.h5 --sorting CSC01/sort_pos_myrun
```

Open the printed URL (default `http://127.0.0.1:8765`).

## Remote / AMAREL (Rutgers)

Install in your conda/venv on the cluster the same way (`pip install -e .`).

Run sorting headlessly on a compute node:

```bash
combinato sort --datafile /scratch/$USER/CSC01/data_CSC01.h5 --label amarel
```

For the GUI over SSH, port-forward from your laptop:

```bash
# on the cluster (login or compute node with the data)
combinato gui --datafile ... --sorting ... --host 127.0.0.1 --port 8765

# on your laptop
ssh -L 8765:localhost:8765 <netid>@amarel.rutgers.edu
```

Then open `http://127.0.0.1:8765` locally.

## Configuration

Defaults live in code (`combinato.config.CombinatoConfig`). Override with:

1. `combinato.yaml` in the working directory, or `$COMBINATO_CONFIG`
2. Environment variables `COMBINATO_<FieldName>` (e.g. `COMBINATO_nFeatures=12`)
3. CLI `--config path.yaml` / `--verbose`

Example `combinato.yaml`:

```yaml
nFeatures: 10
RecursiveDepth: 1
threshold_factor: 5
LogLevel: INFO
```

## CLI reference

| Command | Purpose |
|---------|---------|
| `combinato extract` | Spike detection from `.ncs` / `.mat` / HDF5 |
| `combinato prepare` | Build session folders for clustering |
| `combinato cluster` | Run SPC + template match per session |
| `combinato combine` | Concatenate sessions + group |
| `combinato sort` | prepare → cluster → combine in one go |
| `combinato gui` | Local web curation UI |

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

CI runs on Ubuntu, macOS, and Windows (Python 3.10 / 3.12). SPC-dependent tests skip cleanly if the binary cannot run.

## Relationship to upstream Combinato

Algorithms are ported from Johannes Niediek's Combinato (MIT). Cite the original paper when publishing:

> Niediek et al., PLOS ONE 2016. doi:10.1371/journal.pone.0166598

## License

MIT — see [LICENSE](LICENSE).
