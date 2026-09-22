#!/usr/bin/env python3
"""
dd_final_docking.py

Stage VIII (final phase) of the Deep Docking protocol: take the molecules
kept after the last DD iteration (the output of the DD `final_extraction.py`
utility -- `smiles.csv` + `id_score.csv`), select the top-N by DD score,
build 3D structures for them, dock the top-N with a docking engine,
and merge the results into one SDF ready to hand off.

This is the execution engine for final docking, so everything that selects/prepares/docks/merges 
molecules lives here, whether it is invoked by hand, from `dd_final_docking_wizard.py`, 
or from a hand-written SLURM script.

Two ways to configure it
-------------------------
  --campaign-config campaign.yaml
      Reads receptor/box/docking-program/score-keyword straight from an
      existing Deep Docking campaign's config, and reuses that campaign's
      own ligand-prep and AutoDock-GPU export tools (`dd_ligand_prep.py`,
      `dd_autodock_export.py` in the sibling `dd_active_learning/`
      directory) so final docking looks and behaves exactly like every
      other phase of the campaign. This is what the wizard always passes.

  explicit flags (--backend / --receptor / --center / --size / ...)
      For a one-off run, a target that was never run through
      `dd_orchestrator.py`/`campaign.yaml`, or driving each stage by hand.
      Uses this script's own RDKit/OpenEye prepare backends and a
      meeko-on-the-fly PDBQT conversion for AutoDock-GPU.

Either way, every stage after `select` reads/writes a shared, self-describing
directory layout:

    <final-dir>/
      id_score.csv            (top-N, id,score, sorted best-first)
      smile/chunk_00000.smi   (headerless "smiles id", up to --batch-size each)
      smile/chunk_00001.smi
      ...
      sdf/chunk_00000.sdf     (3D, prepared -- gnina reads these directly)
      pdbqt/chunk_00000/*.pdbqt  (per-molecule PDBQT -- AutoDock-GPU reads these)
      docked/chunk_00000_docked.sdf   (one scored SDF per chunk, backend-agnostic)
      ...

Pipeline
--------
1. select  : slice/re-sort {smiles,id_score}.csv -> top-N, chunked into
             <final-dir>/smile/chunk_NNNNN.smi
2. prepare : every chunk's SMILES -> 3D, protonated structures
             (<final-dir>/sdf/ and/or pdbqt/)
3. dock    : dock one chunk (one SLURM array task) with gnina or
             AutoDock-GPU -> <final-dir>/docked/chunk_NNNNN_docked.sdf
4. merge   : collect every chunk's docked output into one score-sorted SDF
             for handoff, plus a summary CSV
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Campaign-config integration helpers
# ---------------------------------------------------------------------------

def _dd_active_learning_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "dd_active_learning"


def _load_campaign_config(path: str) -> dict:
    sys.path.insert(0, str(_dd_active_learning_dir()))
    from dd_utils import load_config
    return load_config(path)


def _campaign_docking_program(cfg: dict) -> str:
    sys.path.insert(0, str(_dd_active_learning_dir()))
    from dd_orchestrator import _docking_program
    return _docking_program(cfg)


def _read_box_json(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    return {"center": d["center"], "size": d["size"]}


def _require_columns(fieldnames, required, path):
    if fieldnames is None or any(c not in fieldnames for c in required):
        raise SystemExit(
            "ERROR: %s does not have the expected columns %s "
            "(found: %s). If your final_extraction output has different "
            "column names, rename the header row or edit cmd_select()." %
            (path, required, fieldnames))


# ---------------------------------------------------------------------------
# Stage 1: select top-N from an existing DD final_extraction.py result
# ---------------------------------------------------------------------------
#
# DD's final_extraction.py (utilities/final_extraction.py in the DD_protocol
# repo) writes two files, already merged and sorted descending by "score"
# (higher score = more likely a true virtual hit):
#   id_score.csv : header "id,score"          (comma-separated)
#   smiles.csv   : header "smile id"          (space-separated)
# in the same row order. If --mols-to-dock was used when running
# final_extraction.py, both files are already truncated to that count.
#
# We do not rely on the files still being in that order (they may have been
# copied, concatenated across array tasks, or hand-edited since). We
# re-merge on id and re-sort by score explicitly. This is a few hundred
# thousand rows, trivial in memory with plain csv.
#
# The selected top-N is written chunked into <final-dir>/smile/chunk_NNNNN.smi
# (headerless "smiles id", the same format dd_ligand_prep.py already expects
# for its --smiles-dir input) rather than one monolithic file, so that every
# later stage -- prepare, dock, merge -- operates on the same chunk
# granularity a SLURM array job needs, and standalone/campaign-integrated
# runs share one convention.

def cmd_select(args):
    id_to_score = {}
    with open(args.id_score, newline="") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, ["id", "score"], args.id_score)
        for row in reader:
            id_to_score[row["id"]] = float(row["score"])

    rows = []
    with open(args.smiles, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=" ")
        _require_columns(reader.fieldnames, ["smile", "id"], args.smiles)
        for row in reader:
            score = id_to_score.get(row["id"])
            if score is None:
                continue
            rows.append((row["id"], row["smile"], score))

    missing = len(id_to_score) - len(rows)
    if missing > 0:
        print("WARNING: %d ids in %s had no matching SMILES in %s" %
              (missing, args.id_score, args.smiles))

    rows.sort(key=lambda r: r[2], reverse=True)

    top_n = args.top_n
    if top_n > len(rows):
        print("WARNING: requested top-n=%d but only %d molecules available; "
              "using all of them" % (top_n, len(rows)))
        top_n = len(rows)
    selected = rows[:top_n]

    final_dir = Path(args.final_dir)
    smile_dir = final_dir / "smile"
    smile_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    n_chunks = max(1, -(-len(selected) // batch_size)) if selected else 0
    for i in range(n_chunks):
        chunk_rows = selected[i * batch_size:(i + 1) * batch_size]
        chunk_path = smile_dir / ("chunk_%05d.smi" % i)
        with open(chunk_path, "w") as fh:
            for mol_id, smile, _score in chunk_rows:
                fh.write("%s %s\n" % (smile, mol_id))

    score_out = final_dir / "id_score.csv"
    with open(score_out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "score"])
        for mol_id, _smile, score in selected:
            writer.writerow([mol_id, score])

    print("Selected top %d of %d molecules by DD score." %
          (len(selected), len(rows)))
    if selected:
        print("  score range kept: %.6g (best) to %.6g (worst kept)" %
              (selected[0][2], selected[-1][2]))
    print("  wrote %d chunk file(s) of up to %d molecules each -> %s" %
          (n_chunks, batch_size, smile_dir))
    print("  wrote %s" % score_out)


# ---------------------------------------------------------------------------
# Stage 2: prepare 3D structures for the selected subset
# ---------------------------------------------------------------------------
#
# final_extraction.py only carries flat 2D SMILES + id. The campaign's own
# ligand prep (dd_ligand_prep.py, if --campaign-config is given) or this
# script's own RDKit/OpenEye backends (standalone mode) turn each chunk's
# SMILES into 3D, protonated structures.

def cmd_prepare(args):
    if args.campaign_config:
        _prepare_campaign(args)
    elif args.backend == "openeye":
        _prepare_openeye_dir(args)
    else:
        _prepare_rdkit_dir(args)


def _prepare_campaign(args):
    cfg = _load_campaign_config(args.campaign_config)
    program = _campaign_docking_program(cfg)
    fmt = "sdf" if program == "GNINA" else "pdbqt"

    smiles_dir = os.path.join(args.final_dir, "smile")
    script = _dd_active_learning_dir() / "dd_ligand_prep.py"
    cmd = [sys.executable, str(script),
           "--smiles-dir", smiles_dir,
           "--out-dir", args.final_dir,
           "--format", fmt,
           "--nprocs", str(args.nprocs)]
    print("Running: " + " ".join(cmd))
    if args.dry_run:
        print("[dry-run] not executing")
        return
    subprocess.run(cmd, check=True)


def _make_protonator(ph):
    try:
        from dimorphite_dl import DimorphiteDL
        ddl = DimorphiteDL(min_ph=ph, max_ph=ph, pka_precision=0.0)

        def protonate(smiles):
            variants = ddl.protonate(smiles)
            return variants[0] if variants else smiles
        print("Using Dimorphite-DL for protonation at pH %.1f "
              "(this repo's own tautomer step uses OpenEye TAUTOMERS "
              "instead. Use --backend openeye or --campaign-config if "
              "you need an exact match)." % ph)
        return protonate
    except ImportError:
        print("WARNING: dimorphite_dl not importable: writing molecules "
              "at their input protonation state (no pH adjustment). "
              "Install dimorphite-dl, or use --backend openeye / "
              "--campaign-config for a protonation-aware prep.")
        return None


def _prepare_rdkit_dir(args):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    protonate = _make_protonator(args.ph)

    smile_dir = Path(args.final_dir) / "smile"
    chunks = sorted(smile_dir.glob("chunk_*.smi"))
    if not chunks:
        raise SystemExit(
            "ERROR: no chunk_*.smi files in %s -- run 'select' first." %
            smile_dir)

    sdf_dir = Path(args.final_dir) / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    n_in = n_ok = n_embed_fail = 0
    for chunk_path in chunks:
        out_path = sdf_dir / (chunk_path.stem + ".sdf")
        writer = Chem.SDWriter(str(out_path))
        with open(chunk_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                smi, mol_id = line.split(None, 1)
                n_in += 1
                s = protonate(smi) if protonate else smi
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    n_embed_fail += 1
                    continue
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 0xC0FFEE
                if AllChem.EmbedMolecule(mol, params) != 0:
                    n_embed_fail += 1
                    continue
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except ValueError:
                    pass  # some ions/fragments have no MMFF params; keep raw embed
                mol.SetProp("_Name", mol_id)
                mol.SetProp("dd_id", mol_id)
                writer.write(mol)
                n_ok += 1
        writer.close()

    print("Prepared %d / %d molecules (%d failed to parse/embed)." %
          (n_ok, n_in, n_embed_fail))
    print("  wrote %d SDF chunk file(s) -> %s" % (len(chunks), sdf_dir))


def _prepare_openeye_dir(args):
    import gzip

    if shutil.which("oeomega") is None:
        raise SystemExit(
            "ERROR: oeomega not found on PATH. Make sure the "
            "OpenEye bin directory is on PATH and OE_LICENSE is exported "
            "the same way slurm/05_omega.sh does it before running this "
            "stage.")

    smile_dir = Path(args.final_dir) / "smile"
    chunks = sorted(smile_dir.glob("chunk_*.smi"))
    if not chunks:
        raise SystemExit(
            "ERROR: no chunk_*.smi files in %s -- run 'select' first." %
            smile_dir)

    sdf_dir = Path(args.final_dir) / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    # Our chunk files are already headerless "smile id" so no tempfile reformatting step is needed here.
    for chunk_path in chunks:
        out_sdf = sdf_dir / (chunk_path.stem + ".sdf")
        omega_out = str(out_sdf) + ".gz"
        cmd = [
            "oeomega", args.mode,
            "-in", str(chunk_path),
            "-out", omega_out,
            "-maxconfs", str(args.max_confs),
            "-mpi_np", str(args.mpi_np),
            "-strictstereo", "true" if args.strict_stereo else "false",
        ]
        if args.omega_extra_args:
            cmd.extend(args.omega_extra_args.split())
        print("Running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        with gzip.open(omega_out, "rb") as src, open(out_sdf, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(omega_out)

    print("Prepared %d chunk(s) with OMEGA (%s mode, %d conf(s) max) -> %s" %
          (len(chunks), args.mode, args.max_confs, sdf_dir))
    print("Note: OMEGA drops molecules it cannot embed (e.g. under "
          "--strictstereo true with unresolved stereocenters). Compare "
          "molecule counts against the input .smi files if that matters "
          "here.")


# ---------------------------------------------------------------------------
# Stage 3: dock one chunk (one SLURM array task) with a swappable backend
# ---------------------------------------------------------------------------
#
# Adding a new docking engine means writing one more _dock_<engine>_chunk
# function with this signature and one more branch in cmd_dock.

def cmd_dock(args):
    chunk = "chunk_%05d" % args.array_task_id
    final_dir = args.final_dir
    docked_dir = Path(final_dir) / "docked"
    docked_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = str(docked_dir / (chunk + "_docked.sdf"))

    if args.campaign_config:
        cfg = _load_campaign_config(args.campaign_config)
        program = _campaign_docking_program(cfg)
        dock_cfg = cfg["docking"]
        if program == "GNINA":
            box = _read_box_json(dock_cfg["box_json"])
            _dock_gnina_chunk(
                final_dir, chunk,
                receptor=dock_cfg["receptor_file"],
                center=box["center"], size=box["size"],
                autobox_ligand=None,
                cnn=dock_cfg.get("gnina_cnn", "rescore"),
                exhaustiveness=dock_cfg.get("exhaustiveness", 8),
                out_sdf=out_sdf, dry_run=args.dry_run)
        else:
            _dock_autodock_chunk(
                final_dir, chunk,
                maps_fld=dock_cfg["maps_fld"],
                autodock_bin=dock_cfg.get("autodock_bin", "autodock_gpu_128wi"),
                nrun=dock_cfg.get("autodock_nrun", 10),
                out_sdf=out_sdf, dry_run=args.dry_run)
    else:
        if not args.backend:
            raise SystemExit(
                "ERROR: pass --campaign-config campaign.yaml, or --backend "
                "gnina|autodock_gpu plus its required flags.")
        if args.backend == "gnina":
            if not args.receptor:
                raise SystemExit(
                    "ERROR: --backend gnina requires --receptor.")
            _dock_gnina_chunk(
                final_dir, chunk, receptor=args.receptor,
                center=args.center, size=args.size,
                autobox_ligand=args.autobox_ligand,
                cnn=args.cnn_scoring, exhaustiveness=args.exhaustiveness,
                out_sdf=out_sdf, dry_run=args.dry_run)
        else:
            if not args.receptor_fld:
                raise SystemExit(
                    "ERROR: --backend autodock_gpu requires --receptor-fld "
                    "pointing at the receptor's AutoGrid4 .fld/.map files "
                    "(prepare these once per receptor with "
                    "prepare_receptor4.py + autogrid4 before running this "
                    "stage).")
            _dock_autodock_chunk(
                final_dir, chunk, maps_fld=args.receptor_fld,
                autodock_bin=args.autodock_bin, nrun=args.autodock_nrun,
                out_sdf=out_sdf, dry_run=args.dry_run)

    if not args.dry_run:
        print("Task %s: done -> %s" % (chunk, out_sdf))


def _dock_gnina_chunk(final_dir, chunk, receptor, center, size,
                       autobox_ligand, cnn, exhaustiveness, out_sdf,
                       dry_run):
    """Gnina docking for one chunk's prepared SDF (<final-dir>/sdf/<chunk>.sdf),
    into an explicit box or a reference-ligand autobox. Gnina keeps scores on
    the same scale used to train the DD
    classifier during this campaign."""
    sdf_in = os.path.join(final_dir, "sdf", chunk + ".sdf")
    if not os.path.exists(sdf_in):
        raise SystemExit(
            "ERROR: %s not found -- run 'prepare' first (or check "
            "--array-task-id / --final-dir)." % sdf_in)
    if not autobox_ligand and not (center and size):
        raise SystemExit(
            "ERROR: gnina needs a docking box: pass --center X Y Z and "
            "--size X Y Z, --autobox-ligand, or a --campaign-config whose "
            "docking.box_json already has one (see dd_receptor_prep.py).")

    cmd = [
        "gnina",
        "--receptor", receptor,
        "--ligand", sdf_in,
        "--out", out_sdf,
        "--exhaustiveness", str(exhaustiveness),
        "--seed", "0",
    ]
    if autobox_ligand:
        cmd += ["--autobox_ligand", autobox_ligand]
    else:
        cx, cy, cz = center
        sx, sy, sz = size
        cmd += ["--center_x", str(cx), "--center_y", str(cy),
                "--center_z", str(cz), "--size_x", str(sx),
                "--size_y", str(sy), "--size_z", str(sz)]
    if cnn:
        cmd += ["--cnn_scoring", cnn]

    print("Running: " + " ".join(cmd))
    if dry_run:
        print("[dry-run] not executing")
        return
    subprocess.run(cmd, check=True)


def _dock_autodock_chunk(final_dir, chunk, maps_fld, autodock_bin, nrun,
                          out_sdf, dry_run):
    """AutoDock-GPU docking for one chunk: batch every ligand's PDBQT against
    the pre-computed grid maps, then convert the .dlg results to a single
    scored SDF via dd_active_learning/dd_autodock_export.py, so both standalone
    and campaign-integrated final docking end up with an identical
    <chunk>_docked.sdf, regardless of which produced the input PDBQTs."""
    pdbqt_dir = Path(final_dir) / "pdbqt" / chunk
    if not pdbqt_dir.is_dir() or not any(pdbqt_dir.glob("*.pdbqt")):
        # Standalone fallback: --campaign-config runs get pdbqt/<chunk>/
        # from dd_ligand_prep.py's "prepare" step. A standalone run
        # that only asked for an SDF needs it converted here instead, once,
        # with meeko.
        pdbqt_dir = _meeko_convert_chunk(final_dir, chunk)

    docked_scratch = Path(final_dir) / "docked" / (chunk + "_raw")
    docked_scratch.mkdir(parents=True, exist_ok=True)

    pdbqt_files = sorted(pdbqt_dir.glob("*.pdbqt"))
    if not pdbqt_files:
        raise SystemExit("ERROR: no .pdbqt files in %s" % pdbqt_dir)

    # AutoDock-GPU batch filelist: maps .fld on line 1, then (ligand pdbqt,
    # result basename) pairs.
    filelist_path = docked_scratch / "filelist.txt"
    with open(filelist_path, "w") as fh:
        fh.write(str(maps_fld) + "\n")
        for p in pdbqt_files:
            fh.write(str(p) + "\n")
            fh.write(str(docked_scratch / p.stem) + "\n")

    cmd = [autodock_bin, "--filelist", str(filelist_path), "--nrun", str(nrun)]
    print("Running: " + " ".join(cmd))
    if dry_run:
        print("[dry-run] not executing (and skipping dd_autodock_export.py)")
        return
    subprocess.run(cmd, check=True)

    export_script = _dd_active_learning_dir() / "dd_autodock_export.py"
    export_cmd = [sys.executable, str(export_script),
                  "--dlg-dir", str(docked_scratch), "--out-sdf", out_sdf]
    print("Running: " + " ".join(export_cmd))
    subprocess.run(export_cmd, check=True)


def _meeko_convert_chunk(final_dir, chunk):
    """Standalone-mode-only helper: convert one chunk's prepared SDF
    (<final-dir>/sdf/<chunk>.sdf) to per-molecule PDBQT with meeko, since no
    --campaign-config was given to have dd_ligand_prep.py produce PDBQTs
    up front. Not used in campaign mode."""
    sdf_in = os.path.join(final_dir, "sdf", chunk + ".sdf")
    if not os.path.exists(sdf_in):
        raise SystemExit(
            "ERROR: neither %s/pdbqt/%s/ nor %s exist -- run 'prepare' "
            "first." % (final_dir, chunk, sdf_in))
    if shutil.which("mk_prepare_ligand.py") is None:
        raise SystemExit(
            "ERROR: meeko (mk_prepare_ligand.py) not found on PATH. "
            "Install with 'pip install meeko' in the docking venv, or use "
            "--campaign-config so 'prepare' produces PDBQTs up front via "
            "dd_ligand_prep.py.")

    from rdkit import Chem

    out_dir = Path(final_dir) / "pdbqt" / chunk
    out_dir.mkdir(parents=True, exist_ok=True)
    supplier = Chem.SDMolSupplier(sdf_in, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        mol_id = mol.GetProp("dd_id") if mol.HasProp("dd_id") else \
            mol.GetProp("_Name")
        lig_sdf = out_dir / (mol_id + ".sdf")
        w = Chem.SDWriter(str(lig_sdf))
        w.write(mol)
        w.close()
        subprocess.run(
            ["mk_prepare_ligand.py", "-i", str(lig_sdf),
             "-o", str(out_dir / (mol_id + ".pdbqt"))],
            check=True)
        lig_sdf.unlink()

    print("Converted chunk %s to PDBQT via meeko -> %s" % (chunk, out_dir))
    return out_dir


# ---------------------------------------------------------------------------
# Stage 4: merge all per-chunk docking results into one handoff SDF
# ---------------------------------------------------------------------------
#
# Backend-agnostic: both _dock_gnina_chunk and _dock_autodock_chunk always
# normalize their output to <final-dir>/docked/<chunk>_docked.sdf with the
# score as an SD tag, so merge never needs to know which engine produced it.

def _best_pose_per_molecule(mols, score_tag, lower_is_better):
    best = {}
    for mol in mols:
        if mol is None or not mol.HasProp(score_tag):
            continue
        mol_id = mol.GetProp("dd_id") if mol.HasProp("dd_id") else \
            mol.GetProp("_Name")
        score = float(mol.GetProp(score_tag))
        current = best.get(mol_id)
        if current is None:
            best[mol_id] = (score, mol)
            continue
        better = score < current[0] if lower_is_better else score > current[0]
        if better:
            best[mol_id] = (score, mol)
    return best


def cmd_merge(args):
    from rdkit import Chem

    score_tag = args.score_tag
    if args.campaign_config and score_tag is None:
        cfg = _load_campaign_config(args.campaign_config)
        score_tag = cfg["docking"]["score_keyword"]
    if score_tag is None:
        score_tag = "minimizedAffinity"
    lower_is_better = args.lower_is_better

    docked_dir = Path(args.final_dir) / "docked"
    chunk_paths = sorted(docked_dir.glob("*_docked.sdf"))
    if not chunk_paths:
        raise SystemExit(
            "ERROR: no *_docked.sdf files in %s. Run 'dock' for every "
            "array task first." % docked_dir)

    all_mols = []
    for path in chunk_paths:
        all_mols.extend(list(Chem.SDMolSupplier(str(path), removeHs=False)))

    best = _best_pose_per_molecule(all_mols, score_tag, lower_is_better)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0],
                     reverse=not lower_is_better)

    writer = Chem.SDWriter(args.out)
    with open(args.summary_csv, "w", newline="") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["id", score_tag])
        for mol_id, (score, mol) in ranked:
            mol.SetProp(score_tag, str(score))
            writer.write(mol)
            csv_writer.writerow([mol_id, score])
    writer.close()

    direction = "ascending (lower = better)" if lower_is_better \
        else "descending (higher = better)"
    print("Merged %d docked molecules from %d chunk file(s)." %
          (len(ranked), len(chunk_paths)))
    print("  ranked by %s, %s" % (score_tag, direction))
    if ranked:
        print("  best: %s = %.4g" % (ranked[0][0], ranked[0][1][0]))
        print("  worst kept: %s = %.4g" % (ranked[-1][0], ranked[-1][1][0]))
    print("  wrote %s" % args.out)
    print("  wrote %s" % args.summary_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("select", help="slice/re-sort a final_extraction "
                                        "result to the top-N molecules, "
                                        "chunked for the later stages")
    ps.add_argument("--smiles", required=True,
                     help="final_extraction.py smiles.csv (space-separated: "
                          "smile id)")
    ps.add_argument("--id-score", required=True,
                     help="final_extraction.py id_score.csv (comma-separated: "
                          "id,score)")
    ps.add_argument("--top-n", type=int, required=True)
    ps.add_argument("--final-dir", required=True,
                     help="output directory; writes id_score.csv and "
                          "smile/chunk_NNNNN.smi here")
    ps.add_argument("--batch-size", type=int, default=10000,
                     help="molecules per chunk file (default: 10000, sized "
                          "for one 24h docking array task)")
    ps.set_defaults(func=cmd_select)

    pp = sub.add_parser("prepare", help="build 3D, protonated structures "
                                         "for every chunk in --final-dir")
    pp.add_argument("--final-dir", required=True,
                     help="output of the select stage")
    pp.add_argument("--campaign-config",
                     help="campaign.yaml, if given, shells out to "
                          "dd_active_learning/dd_ligand_prep.py using this "
                          "campaign's docking.program to pick sdf vs pdbqt "
                          "output, and every --backend/--ph/... flag below "
                          "is ignored")
    pp.add_argument("--backend", choices=["rdkit", "openeye"],
                     default="rdkit",
                     help="standalone mode only (no --campaign-config). "
                          "rdkit: self-contained (RDKit + optional "
                          "Dimorphite-DL), no OpenEye license needed. "
                          "openeye: calls oeomega directly, same tool/"
                          "flags as dd_prep's OmegaStep.")
    pp.add_argument("--ph", type=float, default=7.4,
                     help="standalone --backend rdkit only (Dimorphite-DL "
                          "protonation)")
    pp.add_argument("--mode", choices=["classic", "pose"], default="classic",
                     help="standalone --backend openeye only, matches "
                          "OmegaConfig.mode")
    pp.add_argument("--max-confs", type=int, default=1,
                     help="standalone --backend openeye only, matches "
                          "OmegaConfig.max_confs")
    pp.add_argument("--mpi-np", type=int, default=8,
                     help="standalone --backend openeye only, matches "
                          "OmegaConfig.mpi_np")
    pp.add_argument("--strict-stereo", action="store_true", default=False,
                     help="standalone --backend openeye only, matches "
                          "OmegaConfig.strict_stereo")
    pp.add_argument("--omega-extra-args", default="",
                     help="standalone --backend openeye only, extra "
                          "oeomega flags passed verbatim, matches "
                          "OmegaConfig.extra_args")
    pp.add_argument("--nprocs", type=int, default=1,
                     help="--campaign-config only: worker processes passed "
                          "to dd_ligand_prep.py")
    pp.add_argument("--dry-run", action="store_true", default=False,
                     help="print the command that would run instead of "
                          "executing it (--campaign-config only)")
    pp.set_defaults(func=cmd_prepare)

    pd_ = sub.add_parser("dock", help="dock one chunk (run as a SLURM "
                                       "array task, one per chunk)")
    pd_.add_argument("--final-dir", required=True,
                      help="output of the select/prepare stages")
    pd_.add_argument("--array-task-id", type=int, required=True,
                      help="0-based; selects chunk_<task-id>.sdf/pdbqt -- "
                           "pass $SLURM_ARRAY_TASK_ID")
    pd_.add_argument("--campaign-config",
                      help="campaign.yaml, if given, reads receptor, box, "
                           "docking program and its options from it, and "
                           "every --backend/--receptor/... flag below is "
                           "ignored")
    pd_.add_argument("--backend", choices=["gnina", "autodock_gpu"],
                      help="standalone mode only (no --campaign-config)")
    # gnina-specific (standalone mode)
    pd_.add_argument("--receptor",
                      help="standalone --backend gnina: prepared receptor "
                           "(PDBQT)")
    pd_.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    pd_.add_argument("--size", type=float, nargs=3, metavar=("X", "Y", "Z"))
    pd_.add_argument("--autobox-ligand",
                      help="standalone --backend gnina: use a reference "
                           "ligand's bounding box instead of --center/--size")
    pd_.add_argument("--exhaustiveness", type=int, default=8)
    pd_.add_argument("--cnn_scoring",
                      choices=["none", "rescore", "refinement", "all"])
    # autodock_gpu-specific (standalone mode)
    pd_.add_argument("--receptor-fld",
                      help="standalone --backend autodock_gpu: receptor "
                           ".fld file from AutoGrid4")
    pd_.add_argument("--autodock-bin", default="autodock_gpu_128wi",
                      help="standalone --backend autodock_gpu: binary name "
                           "for your GPU build")
    pd_.add_argument("--autodock-nrun", type=int, default=10,
                      help="standalone --backend autodock_gpu: docking runs "
                           "per ligand")
    pd_.add_argument("--dry-run", action="store_true", default=False,
                      help="print the docking command instead of running it")
    pd_.set_defaults(func=cmd_dock)

    pm = sub.add_parser("merge", help="collect every chunk's docked output "
                                       "into one handoff SDF, best pose per "
                                       "molecule")
    pm.add_argument("--final-dir", required=True,
                     help="the --final-dir used in the select/dock stages")
    pm.add_argument("--campaign-config",
                     help="campaign.yaml -- if given and --score-tag is "
                          "not passed, defaults --score-tag from this "
                          "campaign's docking.score_keyword")
    pm.add_argument("--score-tag", default=None,
                     help="SD tag to rank by (gnina: minimizedAffinity, "
                          "CNNscore, CNNaffinity; autodock_gpu: "
                          "ADGPU_score). Default: minimizedAffinity, or "
                          "--campaign-config's docking.score_keyword")
    pm.add_argument("--lower-is-better", action="store_true", default=True)
    pm.add_argument("--higher-is-better", dest="lower_is_better",
                     action="store_false",
                     help="use for CNNscore/CNNaffinity, where higher = "
                          "better")
    pm.add_argument("--out", required=True)
    pm.add_argument("--summary-csv", required=True)
    pm.set_defaults(func=cmd_merge)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
