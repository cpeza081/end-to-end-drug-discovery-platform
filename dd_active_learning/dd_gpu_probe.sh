#!/bin/bash
# =============================================================================
# dd_gpu_probe.sh - find faulty GPU nodes on a SLURM cluster (e.g. Fir).
# =============================================================================
# Two passes:
#   1) PASSIVE: list nodes SLURM already flags as drained/down + the reason.
#      Works any time (even during maintenance) and needs no allocation - this
#      is usually where the known-bad GPUs already are.
#   2) ACTIVE : pin a short `nvidia-smi` job to each currently-AVAILABLE GPU
#      node and report which ones fail. Catches GPUs that are UP but broken and
#      not yet drained.
#
# Usage:
#   ./dd_gpu_probe.sh <account> <gpu_type> [--active]
# Example:
#   ./dd_gpu_probe.sh rrg-checco89 nvidia_h100_80gb_hbm3_1g.10gb --active
#
# Prints a ready-to-paste  exclude_nodes: "..."  line for campaign.yaml.
# Note: the ACTIVE pass reports a node "BAD?" if nvidia-smi fails or the node
# would not free up within the timeout. Verify a "BAD?" against `sinfo` before
# trusting it (a busy node is not necessarily faulty).
# =============================================================================
set -uo pipefail

ACCOUNT="${1:-}"
GPU="${2:-}"
ACTIVE=0
[ "${3:-}" = "--active" ] && ACTIVE=1

if [ -z "$ACCOUNT" ] || [ -z "$GPU" ]; then
    echo "usage: $0 <account> <gpu_type> [--active]" >&2
    echo "  e.g. $0 rrg-checco89 nvidia_h100_80gb_hbm3_1g.10gb --active" >&2
    exit 1
fi

bad_nodes=()

echo "=============================================================="
echo " 1) Nodes SLURM already flags as drained / down (with reason)"
echo "=============================================================="
# %n node, %t state, %E reason. Only non-idle/allocatable states appear in -R.
sinfo -R -o "%n | %t | %E" 2>/dev/null | sort -u | while IFS= read -r line; do
    echo "  $line"
done
# Collect the flagged GPU node names for the exclude list.
while read -r n; do
    [ -n "$n" ] && bad_nodes+=("$n")
done < <(sinfo -h -N -o "%N %t %G" 2>/dev/null \
         | awk '$3 ~ /gpu:/ && ($2 ~ /drain/ || $2 ~ /down/ || $2 ~ /fail/){print $1}' \
         | sort -u)

if [ "$ACTIVE" -eq 1 ]; then
    echo
    echo "=============================================================="
    echo " 2) Actively probing AVAILABLE GPU nodes with nvidia-smi"
    echo "    (account=$ACCOUNT  gpu=$GPU)"
    echo "=============================================================="
    # Only idle/mix (schedulable) GPU nodes; skip drain/down/reserved so a busy
    # node is less likely to masquerade as faulty.
    mapfile -t AVAIL < <(sinfo -h -N -o "%N %t %G" 2>/dev/null \
        | awk '$3 ~ /gpu:/ && ($2=="idle" || $2=="mix"){print $1}' | sort -u)

    if [ "${#AVAIL[@]}" -eq 0 ]; then
        echo "  No idle/mix GPU nodes to probe right now (cluster busy or in maintenance)."
    else
        echo "  Probing ${#AVAIL[@]} node(s); ~2 min each..."
        for n in "${AVAIL[@]}"; do
            if timeout 160 srun --account="$ACCOUNT" --nodelist="$n" \
                 --gres="gpu:${GPU}:1" --time=0:02:00 --mem=4G \
                 --job-name=gpuprobe \
                 bash -c 'nvidia-smi -L && nvidia-smi >/dev/null' >/dev/null 2>&1; then
                echo "  OK    $n"
            else
                echo "  BAD?  $n   (nvidia-smi failed or node did not free up. Verify)"
                bad_nodes+=("$n")
            fi
        done
    fi
fi

# Deduplicate and emit a ready-to-paste exclude list.
echo
echo "=============================================================="
if [ "${#bad_nodes[@]}" -eq 0 ]; then
    echo " No faulty GPU nodes found."
else
    uniq_bad=$(printf '%s\n' "${bad_nodes[@]}" | sort -u | paste -sd, -)
    echo " Suspect nodes -> add this to campaign.yaml under scheduler:"
    echo
    echo "   exclude_nodes: \"$uniq_bad\""
fi
echo "=============================================================="
