#!/usr/bin/env bash

# ===============================
# Global parameters
# ===============================
RECORDCOUNT=25000000
OPCOUNT=3000000   # huge so time, not count, is the limiter
FIELDCOUNT=10
FIELDLEN=200
DBDIR=/tmp/pebble

# This script assumes you've already loaded once with:
# bin/go-ycsb load pebble -P workloads/workloada \
#   -p recordcount=${RECORDCOUNT} -p fieldcount=${FIELDCOUNT} -p fieldlength=${FIELDLEN}

run_experiment() {
  local group="$1"      # WorkloadVsDistribution
  local wl="$2"         # A, B, or C
  local cache="$3"      # none, lru, lfu, density, detox
  local dist="$4"       # zipfian
  local threads="$5"    # 32
  local cachesize="$6"  # 0 or 2500000
  local timeout_secs=600   # 10 minutes

  # ---------------------------
  # Map cache to DB name
  # ---------------------------
  local db=""
  case "$cache" in
    none)    db="pebble" ;;
    lru)     db="pebblelru" ;;
    lfu)     db="pebblelfu" ;;
    density) db="pebbledensity" ;;
    detox)   db="pebbledetox" ;;
    *) echo "Unknown cache type: $cache" >&2; return 1 ;;
  esac

  # ---------------------------
  # Map workload letter to file
  # ---------------------------
  local wlfile=""
  case "$wl" in
    A) wlfile="workloads/workloada" ;;
    B) wlfile="workloads/workloadb" ;;
    C) wlfile="workloads/workloadc" ;;
    *) echo "Unknown workload: $wl" >&2; return 1 ;;
  esac

  # ---------------------------
  # Output file name
  # ---------------------------
  local outfile="${group}_wl${wl}_${cache}_${dist}_t${threads}_ASYNC.out"

  echo "Running ${group} wl${wl} cache=${cache} dist=${dist} threads=${threads} -> ${outfile}"

  # ---------------------------
  # Base go-ycsb run command
  # ---------------------------
  cmd="../bin/go-ycsb run ${db} \
    -P ${wlfile} \
    --threads ${threads} \
    -p recordcount=${RECORDCOUNT} \
    -p operationcount=${OPCOUNT} \
    -p fieldcount=${FIELDCOUNT} \
    -p fieldlength=${FIELDLEN} \
    -p pebble.dir=${DBDIR} \
    -p dropdata=false \
    -p requestdistribution=${dist}"

  # (no hotspot case needed since we only use zipfian)

  # ---------------------------
  # Set cache capacity for cached variants
  # ---------------------------
  if [ "${cachesize}" -gt 0 ]; then
    case "$cache" in
      lru)
        cmd="${cmd} -p pebble.lru_capacity=${cachesize}"
        ;;
      lfu)
        cmd="${cmd} -p pebble.lfu_capacity=${cachesize}"
        ;;
      density)
        cmd="${cmd} -p pebble.density_capacity=${cachesize}"
        ;;
      detox)
        cmd="${cmd} -p pebble.detox_capacity=${cachesize}"
        ;;
    esac
  fi

  # ---------------------------
  # Run with timeout (SIGINT ~ Ctrl+C after 10 minutes)
  # ---------------------------
  if ! eval "timeout -s INT ${timeout_secs} ${cmd} > ${outfile} 2>&1"; then
    rc=$?
    if [ $rc -eq 124 ]; then
      echo "  -> Timed out after ${timeout_secs}s, moving on..."
    else
      echo "  -> Command exited with status ${rc} (see ${outfile})"
    fi
  fi
}

# ======================================
# WorkloadVsDistribution (zipfian only)
# ======================================
# Workloads: A, B, C
# Caches: None, LRU, LFU, Density, DeToX
# Threads: 32
# Distribution: zipfian
# Cache size: 0 for None, 2,500,000 for others

for wl in A B C; do
  dist="zipfian"
  threads=32

  # No cache
  run_experiment "WorkloadVsDistribution" "$wl" "none"    "$dist" $threads 0

  # Cached variants
  run_experiment "WorkloadVsDistribution" "$wl" "lru"     "$dist" $threads 2500000
  run_experiment "WorkloadVsDistribution" "$wl" "lfu"     "$dist" $threads 2500000
  run_experiment "WorkloadVsDistribution" "$wl" "density" "$dist" $threads 2500000
  run_experiment "WorkloadVsDistribution" "$wl" "detox"   "$dist" $threads 2500000
done
