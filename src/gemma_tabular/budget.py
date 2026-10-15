"""Atomic conservative reservations, not an account billing enforcement mechanism."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid

# Rates rechecked on https://modal.com/pricing on 2026-10-15. Recheck before launches.
GPU_RATES = {"L4": 0.000222, "L40S": 0.000542, "A100-80GB": 0.000694}
CPU_RATE = 0.0000131
MEMORY_RATE = 0.00000222
MAX_TOTAL = 30.0
RESERVE_OVERHEAD = 5.0


def worst_case_cost(gpu=None, seconds=600, startup_seconds=300, cpu=4, memory_gib=32):
    if seconds <= 0 or startup_seconds < 0 or cpu <= 0 or memory_gib <= 0:
        raise ValueError("Resource bounds must be positive")
    if gpu is not None and gpu not in GPU_RATES:
        raise ValueError("GPU has no verified price in this ledger")
    rate = GPU_RATES.get(gpu, 0) + cpu * CPU_RATE + memory_gib * MEMORY_RATE
    return math.ceil(rate * (seconds + startup_seconds + 60) * 1.2 * 100) / 100


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, created REAL, name TEXT, maximum REAL, status TEXT, details TEXT)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def reserve(self, name, maximum, details=None):
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Reservation must be a finite positive amount")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            allocated = connection.execute("SELECT COALESCE(SUM(maximum), 0) FROM reservations").fetchone()[0]
            if allocated + maximum > MAX_TOTAL - RESERVE_OVERHEAD:
                raise ValueError("Launch would exceed the $25 job reservation ceiling")
            pending = connection.execute("SELECT COUNT(*) FROM reservations WHERE status = 'reserved'").fetchone()[0]
            if pending:
                raise ValueError("An unresolved reservation exists; reconcile it before another launch")
            identifier = uuid.uuid4().hex
            connection.execute("INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)",
                               (identifier, time.time(), name, maximum, "reserved", json.dumps(details or {})))
        return identifier

    def complete(self, identifier, outcome):
        # Keep the entire allocation after completion. No automatic refunds based on estimated spend.
        with self.connect() as connection:
            changed = connection.execute("UPDATE reservations SET status = ? WHERE id = ? AND status = 'reserved'",
                                         (outcome, identifier)).rowcount
            if changed != 1:
                raise ValueError("Unknown or already reconciled reservation")

    def summary(self):
        with self.connect() as connection:
            rows = connection.execute("SELECT id, name, maximum, status FROM reservations ORDER BY created").fetchall()
        allocated = sum(row[2] for row in rows)
        return {"authorized_total_usd": MAX_TOTAL, "overhead_reserve_usd": RESERVE_OVERHEAD,
                "allocated_worst_case_usd": allocated,
                "unallocated_job_budget_usd": round(MAX_TOTAL - RESERVE_OVERHEAD - allocated, 2),
                "actual_billed_spend_usd": None,
                "reservations": [dict(zip(("id", "name", "maximum_usd", "status"), row)) for row in rows]}
