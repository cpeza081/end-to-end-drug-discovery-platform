#!/usr/bin/env python3
"""
dd_receptor_prep.py
===================
One-time receptor / binding-site preparation for a Deep Docking campaign.

This script performs DD setup automatically and writes a single ``receptor_box.json`` that
BOTH docking engines consume identically:

    * GNINA        -> docks with an explicit box (center/size from the JSON)
    * AUTODOCK_GPU -> grid maps (.maps.fld) are pre-computed on that same box

Binding-site strategies (pluggable)
-----------------------------------
The way the box is derived is a strategy, chosen by ``docking.site.method``.
Strategies are registered in SITE_METHODS. Adding a new one is a
decorated function that returns (center, size).

Built in:
    reference_ligand   Box envelops a known ligand in the pocket (holo
                       structure / co-crystallised ligand) plus padding.
                       Most accurate when a bound ligand is available.
    p2rank             Box centred on the top pocket predicted by P2Rank
                       from the protein alone (works on apo structures,
                       no reference ligand required).
    manual             Explicit box the user already knows: center [x,y,z]
                       and size [x,y,z] taken straight from the config.

Usage
-----
  # Drive everything from the campaign config (recommended, minimal input):
  python dd_receptor_prep.py --config campaign.yaml

  # Or pass options explicitly:
  python dd_receptor_prep.py --program AUTODOCK_GPU \\
      --receptor receptor.pdb --out-dir "$SCRATCH/receptor" \\
      --site-method reference_ligand --reference-ligand ref.sdf --padding 4.0

  # Preview external commands without running them:
  python dd_receptor_prep.py --config campaign.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# load_config lives alongside this script; import works whether run from the
# package directory or elsewhere on PYTHONPATH.
try:
    from dd_utils import load_config
except ImportError:  # pragma: no cover - fallback when not on path
    load_config = None


# ===========================================================================
# Binding-site strategy registry  (extensible)
# ===========================================================================
SITE_METHODS = {}


def site_method(name):
    """Register a binding-site strategy under ``name``.

    A strategy has signature ``fn(site_cfg: dict, receptor: Path,
    dry_run: bool) -> (center, size)`` where center/size are 3-tuples of
    floats in Angstrom.
    """
    def decorator(fn):
        SITE_METHODS[name] = fn
        return fn
    return decorator


# ---- shared geometry ------------------------------------------------------
def _read_ligand_coords(path: Path):
    """Heavy-atom coordinates of a reference ligand (SDF/MOL/MOL2/PDB)."""
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix in (".sdf", ".mol"):
        supplier = Chem.SDMolSupplier(str(path), removeHs=True)
        mol = next((m for m in supplier if m is not None), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=True)
    elif suffix in (".pdb", ".ent"):
        mol = Chem.MolFromPDBFile(str(path), removeHs=True)
    else:
        raise ValueError(f"Unsupported reference-ligand format: {suffix}")

    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read 3D coordinates from {path}")

    conf = mol.GetConformer()
    coords = [
        (conf.GetAtomPosition(a.GetIdx()).x,
         conf.GetAtomPosition(a.GetIdx()).y,
         conf.GetAtomPosition(a.GetIdx()).z)
        for a in mol.GetAtoms() if a.GetAtomicNum() != 1
    ]
    if not coords:
        raise ValueError(f"No heavy atoms found in {path}")
    return coords


def _box_from_coords(coords, padding: float):
    """Center + size of a box enveloping coords, padded on all sides."""
    xs, ys, zs = zip(*coords)
    center = ((min(xs) + max(xs)) / 2.0,
              (min(ys) + max(ys)) / 2.0,
              (min(zs) + max(zs)) / 2.0)
    size = ((max(xs) - min(xs)) + 2 * padding,
            (max(ys) - min(ys)) + 2 * padding,
            (max(zs) - min(zs)) + 2 * padding)
    return center, size


# ---- strategy: reference ligand ------------------------------------------
@site_method("reference_ligand")
def _site_reference_ligand(site_cfg, receptor, dry_run):
    ref = site_cfg.get("reference_ligand")
    if not ref:
        sys.exit("ERROR: site.method=reference_ligand requires "
                 "docking.site.reference_ligand (or --reference-ligand).")
    ref = Path(ref)
    if not ref.is_file():
        sys.exit(f"ERROR: reference ligand not found: {ref}")
    padding = float(site_cfg.get("padding", 4.0))
    print(f"  [reference_ligand] pocket from {ref}  (padding {padding} A)")
    return _box_from_coords(_read_ligand_coords(ref), padding)


# ---- strategy: manual / explicit box -------------------------------------
@site_method("manual")
def _site_manual(site_cfg, receptor, dry_run):
    """Use an explicit box the user already knows.

    Requires ``center: [x, y, z]`` and ``size: [x, y, z]`` (Angstrom) under
    docking.site.  No receptor structure or reference ligand is consulted, so
    this is the fastest path when the pocket box is already known.
    """
    center = site_cfg.get("center")
    size = site_cfg.get("size")
    if center is None or size is None:
        sys.exit("ERROR: site.method=manual requires both docking.site.center "
                 "[x,y,z] and docking.site.size [x,y,z].")
    try:
        center = tuple(float(v) for v in center)
        size = tuple(float(v) for v in size)
    except (TypeError, ValueError):
        sys.exit("ERROR: site.center / site.size must each be three numbers.")
    if len(center) != 3 or len(size) != 3:
        sys.exit("ERROR: site.center and site.size must each have exactly "
                 "three values (x, y, z).")
    if any(v <= 0 for v in size):
        sys.exit("ERROR: site.size values must be positive.")
    print(f"  [manual] explicit box from config")
    return center, size


# ---- strategy: P2Rank pocket prediction ----------------------------------
@site_method("p2rank")
def _site_p2rank(site_cfg, receptor, dry_run):
    """Predict pockets with P2Rank and box the top-ranked one.

    P2Rank runs on the protein alone, so no reference ligand is needed.
    """
    exe = site_cfg.get("p2rank_exec", "prank")
    rank = int(site_cfg.get("pocket_rank", 1))
    box_size = site_cfg.get("box_size", [24.0, 24.0, 24.0])
    if isinstance(box_size, (int, float)):
        box_size = [float(box_size)] * 3
    out_dir = receptor.parent / "p2rank_out"

    if not dry_run and shutil.which(exe) is None:
        sys.exit(f"ERROR: P2Rank executable {exe!r} not found on PATH.")

    _run([exe, "predict", "-f", str(receptor), "-o", str(out_dir)], dry_run)

    center = _parse_p2rank_center(out_dir, receptor, rank, dry_run)
    print(f"  [p2rank] pocket rank {rank} center "
          f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}; "
          f"box {box_size}")
    return center, tuple(float(v) for v in box_size)


def _parse_p2rank_center(out_dir: Path, receptor: Path, rank: int,
                         dry_run: bool):
    """Read the center of the requested pocket from P2Rank's predictions CSV."""
    csv_path = out_dir / f"{receptor.name}_predictions.csv"
    if dry_run and not csv_path.is_file():
        return (0.0, 0.0, 0.0)  # placeholder for --dry-run previews
    if not csv_path.is_file():
        sys.exit(f"ERROR: P2Rank predictions not found: {csv_path}")

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        rows = [{k.strip(): (v.strip() if v is not None else v)
                 for k, v in row.items()} for row in reader]
    if not rows:
        sys.exit(f"ERROR: P2Rank found no pockets in {csv_path}")
    if rank > len(rows):
        sys.exit(f"ERROR: requested pocket rank {rank} but P2Rank found "
                 f"only {len(rows)} pocket(s).")
    row = rows[rank - 1]
    return (float(row["center_x"]), float(row["center_y"]),
            float(row["center_z"]))


# ===========================================================================
# External-tool orchestration
# ===========================================================================
def _run(cmd, dry_run: bool):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def prepare_autodock_maps(receptor: Path, out_dir: Path, center, size,
                          dry_run: bool) -> Path:
    """Build receptor PDBQT + GPF (Meeko) then run AutoGrid -> .maps.fld."""
    name = receptor.stem
    gpf = out_dir / f"{name}.gpf"
    fld = out_dir / f"{name}.maps.fld"

    mk = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    autogrid = shutil.which("autogrid4")
    if not dry_run:
        if mk is None:
            sys.exit("ERROR: mk_prepare_receptor(.py) not found on PATH "
                     "(install Meeko + AutoDock utilities).")
        if autogrid is None:
            sys.exit("ERROR: autogrid4 not found on PATH (install AutoDock 4).")

    cx, cy, cz = center
    sx, sy, sz = size
    _run([mk or "mk_prepare_receptor.py",
          "--read_pdb", str(receptor),
          "-o", str(out_dir / name),
          "-p", "-g",
          "--box_size", f"{sx:.3f}", f"{sy:.3f}", f"{sz:.3f}",
          "--box_center", f"{cx:.3f}", f"{cy:.3f}", f"{cz:.3f}"], dry_run)
    _run([autogrid or "autogrid4", "-p", str(gpf),
          "-l", str(out_dir / f"{name}.glg")], dry_run)
    return fld


# ===========================================================================
# CLI
# ===========================================================================
def _resolve_settings(args) -> dict:
    """Merge campaign-config docking settings with explicit CLI overrides."""
    s = {"site": {}}
    if args.config:
        if load_config is None:
            sys.exit("ERROR: cannot import load_config to read --config.")
        dock = load_config(args.config).get("docking", {})
        s.update({k: v for k, v in dock.items() if k != "site"})
        s["site"] = dict(dock.get("site", {}))

    # CLI overrides win over config.
    if args.program:        s["program"] = args.program
    if args.receptor:       s["receptor_file"] = args.receptor
    if args.site_method:    s["site"]["method"] = args.site_method
    if args.reference_ligand: s["site"]["reference_ligand"] = args.reference_ligand
    if args.padding is not None: s["site"]["padding"] = args.padding
    return s


def main():
    parser = argparse.ArgumentParser(
        description="One-time receptor / binding-site prep for Deep Docking"
    )
    parser.add_argument("--config", help="campaign.yaml to read docking.* from")
    parser.add_argument("--program", choices=("GNINA", "AUTODOCK_GPU"))
    parser.add_argument("--receptor", help="Receptor structure (PDB)")
    parser.add_argument("--out-dir", help="Output dir (default: receptor's dir)")
    parser.add_argument("--site-method", choices=tuple(SITE_METHODS))
    parser.add_argument("--reference-ligand")
    parser.add_argument("--padding", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    s = _resolve_settings(args)
    program = (s.get("program") or "GNINA").upper().replace("-", "_")
    if program in ("AUTODOCKGPU", "ADGPU"):
        program = "AUTODOCK_GPU"

    receptor = s.get("receptor_file")
    if not receptor:
        sys.exit("ERROR: no receptor given (docking.receptor_file or --receptor).")
    receptor = Path(receptor)
    if not receptor.is_file():
        sys.exit(f"ERROR: receptor not found: {receptor}")

    out_dir = Path(args.out_dir) if args.out_dir else receptor.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    method = s["site"].get("method", "reference_ligand")
    if method not in SITE_METHODS:
        sys.exit(f"ERROR: unknown site.method {method!r}. "
                 f"Available: {', '.join(SITE_METHODS)}.")

    print(f"Binding-site strategy: {method}")
    center, size = SITE_METHODS[method](s["site"], receptor, args.dry_run)
    print(f"  center (x,y,z): {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}")
    print(f"  size   (x,y,z): {size[0]:.3f} {size[1]:.3f} {size[2]:.3f}")

    box_json = out_dir / "receptor_box.json"
    box_json.write_text(json.dumps({
        "site_method": method,
        "program": program,
        "receptor": str(receptor.resolve()),
        "center": list(center),
        "size": list(size),
    }, indent=2))
    print(f"  wrote {box_json}")

    print("\nSet in campaign.yaml:")
    print(f'  docking.receptor_file: "{receptor.resolve()}"')
    print(f'  docking.box_json:      "{box_json.resolve()}"')

    if program == "AUTODOCK_GPU":
        print("\nBuilding AutoDock-GPU grid maps (one-time):")
        fld = prepare_autodock_maps(receptor, out_dir, center, size, args.dry_run)
        shown = fld.resolve() if not args.dry_run else fld
        print(f"\nDone. Grid map descriptor: {shown}")
        print(f'  docking.maps_fld: "{shown}"')
    else:
        print("\nGnina is ready: it will dock with the explicit box above.")


if __name__ == "__main__":
    main()
