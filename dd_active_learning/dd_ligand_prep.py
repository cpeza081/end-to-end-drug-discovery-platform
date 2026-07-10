#!/usr/bin/env python3
"""
dd_ligand_prep.py
=================
Phase-2 ligand preparation for the Deep Docking active-learning loop.

Open-source pipeline:

    SMILES chunk (.smi)
        - RDKit ETKDGv3 3D embedding + MMFF/UFF minimisation
        - per-chunk 3D SDF                     (input for Gnina)
        - optional Meeko conversion to PDBQT    (input for AutoDock-GPU)

Design notes
------------
* Parallelism is at the *molecule* level, not the chunk level.  DD's
  extracting_smiles.py emits only a handful of large per-set files
  (train / valid / test), each with up to ~1M molecules, so parallelising
  across chunks alone would leave almost all cores idle.  A single worker pool
  streams molecules from each chunk via ``imap_unordered`` (lazy: bounded
  memory regardless of chunk size) and the main process writes results as they
  arrive.
* One output SDF per input .smi chunk, named ``<chunk>.sdf`` (concatenated mol
  blocks), so the docking phase and DD's extract_labels.py keep seeing the same
  chunk granularity they did under the FRED/GLIDE workflow.
* For AutoDock-GPU we additionally write one PDBQT per molecule into a per-chunk
  sub-directory (``<out_dir>/pdbqt/<chunk>/<molid>.pdbqt``), because
  AutoDock-GPU docks a single ligand per invocation.
* Bad molecules (unparseable SMILES, embedding failures) are logged and
  skipped.
* Work is parallelised across chunks.

Input .smi format (DD convention): whitespace-separated ``SMILES  molecule_id``,
one molecule per line, no header.

Usage
-----
  python dd_ligand_prep.py \\
      --smiles-dir  "$ITER_DIR/smile" \\
      --out-dir     "$ITER_DIR" \\
      --format      sdf            # sdf (Gnina) | pdbqt (AutoDock-GPU) | both
      --nprocs      60
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# How many molecules each worker pulls per task.  Amortises IPC overhead
# without letting many heavy RDKit mols pile up in flight.
_TASK_CHUNKSIZE = 32

# Per-worker state, initialised once in _init_worker (never per molecule).
_WANT_SDF = False
_WANT_PDBQT = False
_MEEKO = None  # (MoleculePreparation, PDBQTWriterLegacy | None)


# ---------------------------------------------------------------------------
# Per-molecule 3D embedding
# ---------------------------------------------------------------------------
def embed_molecule(smiles: str, mol_id: str, seed: int = 0xF00D):
    """
    Parse a SMILES, add hydrogens, embed a single 3D conformer with ETKDGv3
    and minimise it.  Returns an RDKit Mol (``_Name`` set to mol_id) or None
    on any failure.
    """
    # This builds a 3D shape for one molecule so it can be docked later.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    # useRandomCoords helps convergence on awkward/macrocyclic inputs.
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None

    # Prefer MMFF (better geometries); fall back to UFF when MMFF has no
    # parameters for the molecule.  A minimisation failure is non-fatal - the
    # embedded coordinates are still usable for docking.
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    mol.SetProp("_Name", mol_id)
    return mol


def _iter_smiles(smi_path: Path):
    """Yield (smiles, mol_id) pairs from a DD-format .smi chunk."""
    # reads one chunk of molecules line by line.
    with open(smi_path) as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else f"{smi_path.stem}_{idx}"
            yield smiles, mol_id


# ---------------------------------------------------------------------------
# Worker pool (molecule-level)
# ---------------------------------------------------------------------------
def _init_worker(want_sdf: bool, want_pdbqt: bool):
    """Runs once per worker process: silence RDKit and build Meeko once."""
    global _WANT_SDF, _WANT_PDBQT, _MEEKO
    RDLogger.DisableLog("rdApp.*")
    _WANT_SDF = want_sdf
    _WANT_PDBQT = want_pdbqt
    if want_pdbqt:
        from meeko import MoleculePreparation
        try:
            from meeko import PDBQTWriterLegacy  # Meeko >= 0.5
        except ImportError:
            PDBQTWriterLegacy = None
        _MEEKO = (MoleculePreparation(), PDBQTWriterLegacy)


def _mol_to_pdbqt(mol) -> str | None:
    """Convert an embedded RDKit Mol to a single-molecule PDBQT string."""
    prep, writer_legacy = _MEEKO
    try:
        setups = prep.prepare(mol)
    except Exception:
        return None
    if writer_legacy is not None:  # Meeko >= 0.5
        for setup in setups:
            pdbqt_string, ok, _ = writer_legacy.write_string(setup)
            if ok:
                return pdbqt_string
        return None
    try:  # older Meeko
        return prep.write_pdbqt_string()
    except Exception:
        return None


def _prepare_one(task):
    """Worker entry point: embed one molecule, return serialised outputs.

    Returns (mol_id, ok, sdf_text_or_None, pdbqt_text_or_None).  Only plain
    strings cross the process boundary, so memory stays bounded and no RDKit
    pickling quirks apply.  A molecule counts as failed unless every requested
    format was produced.
    """
    smiles, mol_id = task
    mol = embed_molecule(smiles, mol_id)
    if mol is None:
        return (mol_id, False, None, None)

    sdf_text = None
    if _WANT_SDF:
        try:
            sdf_text = Chem.MolToMolBlock(mol)
        except Exception:
            return (mol_id, False, None, None)

    pdbqt_text = None
    if _WANT_PDBQT:
        pdbqt_text = _mol_to_pdbqt(mol)
        if pdbqt_text is None:
            return (mol_id, False, None, None)

    return (mol_id, True, sdf_text, pdbqt_text)


# ---------------------------------------------------------------------------
# Output writing (main process, streaming)
# ---------------------------------------------------------------------------
def _write_result(result, sdf_fh, pdbqt_dir) -> bool:
    """Write one worker result to disk.  Returns True if it was a success."""
    mol_id, ok, sdf_text, pdbqt_text = result
    if not ok:
        return False
    if sdf_fh is not None and sdf_text is not None:
        sdf_fh.write(sdf_text)
        if not sdf_text.endswith("\n"):
            sdf_fh.write("\n")
        sdf_fh.write("$$$$\n")          # SDF record terminator
    if pdbqt_dir is not None and pdbqt_text is not None:
        (pdbqt_dir / f"{mol_id}.pdbqt").write_text(pdbqt_text)
    return True


def _process_chunk(smi_path, out_dir, want_sdf, want_pdbqt, results):
    """Consume an iterator of worker results for one chunk, writing outputs."""
    chunk = smi_path.stem
    sdf_fh = None
    pdbqt_dir = None
    if want_sdf:
        sdf_path = out_dir / "sdf" / f"{chunk}.sdf"
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        sdf_fh = open(sdf_path, "w")
    if want_pdbqt:
        pdbqt_dir = out_dir / "pdbqt" / chunk
        pdbqt_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = 0
    try:
        for result in results:
            if _write_result(result, sdf_fh, pdbqt_dir):
                n_ok += 1
            else:
                n_failed += 1
    finally:
        if sdf_fh is not None:
            sdf_fh.close()
    return n_ok, n_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Open-source Phase-2 ligand prep (RDKit + Meeko) for DD"
    )
    parser.add_argument("--smiles-dir", required=True,
                        help="Directory of .smi chunks (DD 'smile' folder)")
    parser.add_argument("--out-dir", required=True,
                        help="Iteration directory; sdf/ and pdbqt/ are written here")
    parser.add_argument("--format", choices=("sdf", "pdbqt", "both"),
                        default="sdf",
                        help="sdf=Gnina, pdbqt=AutoDock-GPU, both=emit both")
    parser.add_argument("--nprocs", type=int, default=1,
                        help="Number of worker processes")
    args = parser.parse_args()

    smiles_dir = Path(args.smiles_dir)
    out_dir = Path(args.out_dir)
    chunks = sorted(smiles_dir.glob("*.smi"))
    if not chunks:
        print(f"ERROR: no .smi files found in {smiles_dir}", file=sys.stderr)
        sys.exit(1)

    want_sdf = args.format in ("sdf", "both")
    want_pdbqt = args.format in ("pdbqt", "both")
    nprocs = max(1, args.nprocs)

    print(f"Preparing {len(chunks)} chunk(s) from {smiles_dir}")
    print(f"  formats: {'SDF ' if want_sdf else ''}{'PDBQT' if want_pdbqt else ''}")
    print(f"  workers: {nprocs}")

    total_ok = total_failed = 0

    if nprocs == 1:
        # Serial path: no pool, constant memory.
        _init_worker(want_sdf, want_pdbqt)
        for smi_path in chunks:
            results = (_prepare_one(t) for t in _iter_smiles(smi_path))
            ok, failed = _process_chunk(smi_path, out_dir, want_sdf, want_pdbqt, results)
            total_ok += ok
            total_failed += failed
            print(f"  [{smi_path.stem}] ok={ok} failed={failed}")
    else:
        # One pool, reused across chunks; imap_unordered streams molecules so
        # memory stays bounded even for a 1M-molecule chunk.
        ctx = mp.get_context("spawn")
        with ctx.Pool(nprocs, initializer=_init_worker,
                      initargs=(want_sdf, want_pdbqt)) as pool:
            for smi_path in chunks:
                results = pool.imap_unordered(
                    _prepare_one, _iter_smiles(smi_path),
                    chunksize=_TASK_CHUNKSIZE)
                ok, failed = _process_chunk(
                    smi_path, out_dir, want_sdf, want_pdbqt, results)
                total_ok += ok
                total_failed += failed
                print(f"  [{smi_path.stem}] ok={ok} failed={failed}")

    print(f"\nDone. Prepared {total_ok:,} molecules; {total_failed:,} skipped.")
    if total_ok == 0:
        print("ERROR: no molecules were successfully prepared.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
