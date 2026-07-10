#!/usr/bin/env python3
"""
dd_autodock_export.py
====================
Convert a chunk's worth of AutoDock-GPU results (.dlg files, one per ligand)
into a single scored SDF that the Deep Docking label-extraction step can read.

AutoDock-GPU writes a .dlg per ligand. DD's `extract_labels.py` expects one SDF
per chunk with a docking-score field.  This bridges the two:

    <dlg_dir>/*.dlg   -->   <out_sdf>   (best pose per ligand, property ADGPU_score)

The score is AutoDock-GPU's best (lowest) estimated free energy of binding in
kcal/mol i.e. lower is a better binder.  Set `docking.score_keyword: "ADGPU_score"` in campaign.yaml.

Usage
-----
  python dd_autodock_export.py --dlg-dir "$ITER_DIR/docked/chunkX" \\
                               --out-sdf "$ITER_DIR/docked/chunkX_docked.sdf"
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

from meeko import PDBQTMolecule, RDKitMolCreate

RDLogger.DisableLog("rdApp.*")

SCORE_FIELD = "ADGPU_score"


def export_dlg(dlg_path: Path, writer: Chem.SDWriter) -> bool:
    """
    Read one AutoDock-GPU .dlg, take the best-scoring pose, and write it to the
    open SDF writer with the ADGPU_score property.  Returns True on success.
    """
    try:
        pmol = PDBQTMolecule.from_file(str(dlg_path), is_dlg=True,
                                       skip_typing=True)
    except Exception:
        return False

    # pmol.score is the best (lowest) binding energy across docked poses.
    score = getattr(pmol, "score", None)

    rdmols = RDKitMolCreate.from_pdbqt_mol(pmol)
    mol = next((m for m in rdmols if m is not None), None)
    if mol is None:
        return False

    mol.SetProp("_Name", dlg_path.stem)
    if score is not None:
        mol.SetProp(SCORE_FIELD, f"{score:.3f}")
    else:
        # No parsable energy - mark it so extract_labels can drop it rather
        # than silently treating a missing score as a strong binder.
        mol.SetProp(SCORE_FIELD, "nan")

    # SDWriter writes the first conformer (the best pose is pose 0).
    writer.write(mol)
    return True


def main():
    # This starts the export step and gathers the docking files to combine.
    parser = argparse.ArgumentParser(
        description="Export AutoDock-GPU .dlg results to a scored SDF for DD"
    )
    parser.add_argument("--dlg-dir", required=True,
                        help="Directory of .dlg files for one chunk")
    parser.add_argument("--out-sdf", required=True,
                        help="Combined output SDF path")
    args = parser.parse_args()

    dlg_dir = Path(args.dlg_dir)
    # Stream the directory lazily: a chunk can hold ~1M .dlg files, so we never
    # materialise (or sort) the whole listing.  Peek at the first entry to fail
    # fast on an empty directory without creating an empty output SDF.
    dlgs = dlg_dir.glob("*.dlg")
    first = next(dlgs, None)
    if first is None:
        print(f"ERROR: no .dlg files in {dlg_dir}", file=sys.stderr)
        sys.exit(1)

    out_sdf = Path(args.out_sdf)
    out_sdf.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(out_sdf))
    n_ok = n_failed = 0
    for dlg in itertools.chain([first], dlgs):
        if export_dlg(dlg, writer):
            n_ok += 1
        else:
            n_failed += 1
    writer.close()

    print(f"Exported {n_ok} pose(s) to {out_sdf}; {n_failed} failed.")
    if n_ok == 0:
        print("ERROR: no poses exported.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
