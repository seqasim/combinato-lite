"""Typer CLI: combinato extract | prepare | cluster | combine | sort | gui."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .config import load_config
from .logging_config import configure_logging

app = typer.Typer(
    name="combinato",
    help="Combinato-Lite: lightweight, portable spike sorting.",
    no_args_is_help=True,
)


def _bootstrap(config: Optional[Path], verbose: bool):
    overrides = {}
    if verbose:
        overrides["LogLevel"] = "DEBUG"
        overrides["Debug"] = True
    cfg = load_config(path=config, overrides=overrides or None)
    configure_logging(cfg)
    return cfg


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to combinato.yaml"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Combinato-Lite spike sorting."""
    _bootstrap(config, verbose)


@app.command()
def extract(
    files: Optional[list[Path]] = typer.Option(None, "--files", help=".ncs files"),
    matfile: Optional[Path] = typer.Option(None, "--matfile", help="MATLAB .mat file"),
    h5: bool = typer.Option(False, "--h5", help="Treat --files as continuous HDF5"),
    destination: str = typer.Option("", "--destination", "-d"),
    workers: int = typer.Option(4, "--workers", "-j"),
    scale_factor: float = typer.Option(1.0, "--matfile-scale-factor"),
    refscheme: Optional[Path] = typer.Option(
        None, "--refscheme", help="CSV: filename;reference"
    ),
):
    """Extract spikes from .ncs, .mat, or continuous HDF5."""
    from .extract import extract_files, extract_h5, extract_matfile

    if matfile is not None:
        paths = extract_matfile(
            str(matfile), destination=destination, scale_factor=scale_factor
        )
        rprint(f"[green]Wrote[/green] {paths}")
        return

    if not files:
        raise typer.BadParameter("Supply --files or --matfile")

    refs = None
    if refscheme is not None:
        with open(refscheme, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=";")
            refs = {row[0]: row[1] for row in reader if row}

    file_strs = [str(f) for f in files]
    if h5:
        paths = extract_h5(file_strs, destination=destination, n_workers=workers)
    else:
        paths = extract_files(
            file_strs,
            destination=destination,
            n_workers=workers,
            refscheme=refs,
        )
    rprint(f"[green]Wrote[/green] {paths}")


@app.command()
def prepare(
    datafile: Path = typer.Option(..., "--datafile", help="data_*.h5 spike file"),
    neg: bool = typer.Option(False, "--neg", help="Use negative spikes"),
    label: str = typer.Option("sort", "--label"),
    max_nspk: Optional[int] = typer.Option(None, "--max-nspk"),
    start: Optional[int] = typer.Option(None, "--start"),
    stop: Optional[int] = typer.Option(None, "--stop"),
):
    """Create clustering session folders from a spike data file."""
    from .cluster.prepare import prepare_sessions

    sign = "neg" if neg else "pos"
    sessions = prepare_sessions(
        [str(datafile)],
        sign=sign,
        start=start or 0,
        stop=stop,
        max_nspk_session=max_nspk,
        label=label,
        replace=True,
    )
    outfname = f"sort_{sign}_{label}.txt"
    with open(outfname, "a", encoding="utf-8") as outf:
        for name, sgn, ses in sessions:
            outf.write(f"{name} {sgn} {ses}\n")
    rprint(f"[green]Prepared {len(sessions)} sessions[/green] → {outfname}")


@app.command()
def cluster(
    datafile: Optional[Path] = typer.Option(None, "--datafile"),
    sessions: Optional[list[str]] = typer.Option(None, "--sessions"),
    jobs: Optional[Path] = typer.Option(None, "--jobs", help="Job list file"),
    single: bool = typer.Option(False, "--single", help="No multiprocessing"),
    rng: Optional[float] = typer.Option(None, "--rng", help="Random seed"),
):
    """Run SPC clustering on prepared sessions."""
    from .cluster.sort import run_cluster_jobs

    joblist: list[tuple[str, str, str]] = []
    if jobs is not None:
        with open(jobs, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    joblist.append((parts[0], parts[1], parts[2]))
    elif datafile is not None and sessions:
        for ses in sessions:
            sign = "neg" if "neg" in ses else "pos"
            joblist.append((str(datafile), sign, ses))
    else:
        raise typer.BadParameter("Specify --jobs or --datafile and --sessions")

    run_cluster_jobs(joblist, single=single, seed=rng)
    rprint(f"[green]Clustered {len(joblist)} jobs[/green]")


@app.command()
def combine(
    datafile: Path = typer.Option(..., "--datafile"),
    sessions: list[str] = typer.Option(..., "--sessions"),
    label: str = typer.Option(..., "--label"),
    no_grouping: bool = typer.Option(False, "--no-grouping"),
):
    """Concatenate session sortings and optionally group clusters."""
    from .cluster.combine import combine_sessions

    out = combine_sessions(
        str(datafile),
        sessions,
        label,
        do_groups=not no_grouping,
    )
    rprint(f"[green]Combined →[/green] {out}")


@app.command()
def sort(
    datafile: Path = typer.Option(..., "--datafile"),
    neg: bool = typer.Option(False, "--neg"),
    label: str = typer.Option("simple", "--label"),
    rng: Optional[float] = typer.Option(None, "--rng"),
):
    """One-shot: prepare → cluster → combine → group for a single file."""
    from .cluster.combine import combine_sessions
    from .cluster.prepare import prepare_sessions
    from .cluster.sort import run_cluster_jobs

    sign = "neg" if neg else "pos"
    sessions = prepare_sessions(
        [str(datafile)], sign=sign, label=label, replace=True
    )
    if not sessions:
        rprint("[yellow]No spike sessions to sort.[/yellow]")
        raise typer.Exit(0)

    jobs = [(name, sgn, ses) for name, sgn, ses in sessions]
    run_cluster_jobs(jobs, single=True, seed=rng)

    label_full = f"sort_{sign}_{label}"
    ses_names = [os.path.basename(ses) for _, _, ses in sessions]
    out = combine_sessions(str(datafile), ses_names, label_full, do_groups=True)
    rprint(f"[green]Done →[/green] {out}")


@app.command()
def gui(
    datafile: Path = typer.Option(..., "--datafile", help="data_*.h5"),
    sorting: Path = typer.Option(
        ..., "--sorting", help="Path to sort label dir or sort_cat.h5"
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
):
    """Launch the local web curation GUI."""
    from .gui.app import run_gui

    sort_path = sorting
    if sort_path.is_dir():
        sort_path = sort_path / "sort_cat.h5"
    run_gui(str(datafile), str(sort_path), host=host, port=port)


if __name__ == "__main__":
    app()
