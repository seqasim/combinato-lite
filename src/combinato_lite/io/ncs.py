"""Neuralynx .ncs / .nev readers (ported from Combinato nlxio)."""

from __future__ import annotations

import re
from datetime import datetime
from os import stat

import numpy as np

NCS_SAMPLES_PER_REC = 512
NLX_OFFSET = 16 * 1024
NCS_RECSIZE = 1044

TIME_PATTERN = re.compile(r"(\d{1,2}:\d{1,2}:\d{1,2}).(\d{1,3})")

nev_type = np.dtype(
    [
        ("", "V6"),
        ("timestamp", "u8"),
        ("id", "i2"),
        ("nttl", "i2"),
        ("", "V38"),
        ("ev_string", "S128"),
    ]
)

ncs_type = np.dtype(
    [
        ("timestamp", "u8"),
        ("info", ("i4", 3)),
        ("data", ("i2", 512)),
    ]
)


def time_upsample(time, timestep):
    filler = NCS_SAMPLES_PER_REC
    timestep *= 1e6
    base = np.linspace(0, timestep * (filler - 1), filler)
    return np.array([base + x for x in time]).ravel()


def nev_read(filename):
    eventmap = np.memmap(filename, dtype=nev_type, mode="r", offset=NLX_OFFSET)
    return np.array([eventmap["timestamp"], eventmap["nttl"]]).T


def nev_string_read(filename):
    eventmap = np.memmap(filename, dtype=nev_type, mode="r", offset=NLX_OFFSET)
    return np.array([eventmap["timestamp"], eventmap["ev_string"]]).T


class NcsFile:
    """Represents an .ncs file; allows reading data and timestamps."""

    def __init__(self, filename: str):
        self.file = None
        self.filename = filename
        self.num_recs = ncs_num_recs(filename)
        self.header = ncs_info(filename)
        self.file = open(filename, "rb")
        if self.num_recs > 0:
            timestamp = self.read(0, 2, "timestamp")
            self.timestep = float(timestamp[1] - timestamp[0])
            self.timestep /= NCS_SAMPLES_PER_REC * 1e6
        else:
            self.timestep = None

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def read(self, start=0, stop=None, mode="data"):
        if stop is None:
            stop = start + 1
        if stop > start:
            length = stop - start
        else:
            length = 1
        if start + length > self.num_recs + 1:
            raise IOError(
                f"Request to read beyond EOF, filename {self.filename}, "
                f"start {start}, stop {stop}"
            )
        self.file.seek(NLX_OFFSET + start * NCS_RECSIZE)
        data = self.file.read(length * NCS_RECSIZE)
        array_length = int(len(data) / NCS_RECSIZE)
        array_data = np.ndarray(array_length, ncs_type, data)
        if mode == "both":
            return (array_data["data"].flatten(), array_data["timestamp"].flatten())
        if mode in ("data", "timestamp", "info"):
            return array_data[mode].flatten()
        raise ValueError(f"Unknown mode: {mode}")


def ncs_info(filename: str) -> dict:
    """Extract Neuralynx .ncs header fields."""
    d: dict = {}
    with open(filename, "rb") as f:
        header = f.read(NLX_OFFSET)

    for line in header.splitlines():
        try:
            field = [fil.decode() for fil in line.split()]
        except UnicodeDecodeError:
            continue

        if len(field) == 2:
            try:
                field[1] = int(field[1])
            except ValueError:
                try:
                    field[1] = float(field[1])
                except ValueError:
                    pass
            d[field[0][1:]] = field[1]
        elif len(field) == 3:
            if field[0] in ("-TimeCreated", "-TimeClosed"):
                pddate = datetime.strptime(
                    field[1] + " " + field[2], "%Y/%m/%d %H:%M:%S"
                )
                d[field[0][1:]] = pddate
        elif len(field) == 7:
            if field[0] == "##" and field[2] in ("Opened", "Closed"):
                timeg = TIME_PATTERN.match(field[6]).groups()
                pdt = datetime.strptime(field[4] + " " + timeg[0], "%m/%d/%Y %H:%M:%S")
                dt = datetime(
                    pdt.year,
                    pdt.month,
                    pdt.day,
                    pdt.hour,
                    pdt.minute,
                    pdt.second,
                    int(timeg[1]) * 1000,
                )
                d[field[2].lower()] = dt

    if "AcqEntName" not in d and "ADChannel" in d:
        d["AcqEntName"] = "channel" + str(d["ADChannel"])
    return d


def ncs_num_recs(filename: str) -> int:
    data_size = stat(filename).st_size - NLX_OFFSET
    if data_size % NCS_RECSIZE:
        raise Exception(f"{filename} has the wrong size")
    return int(data_size / NCS_RECSIZE)
