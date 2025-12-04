#!/usr/bin/env bash

# ===============================
# Global parameters
# ===============================
RECORDCOUNT=25000000
OPCOUNT=3000000   # huge so time, not count, is the limiter
FIELDCOUNT=10
FIELDLEN=200
DBDIR=/tmp/pebble

run_experiment() {
  local group="$1"      # WorkloadVsDistribution or MultithreadScaling
  local wl="$2"         # A, B, or C
  local cache="$3"      # none, lru, density, detox
  local dist="$4"       # zipfian, uniform, hotspot, sequential
  local threads="$5"    # 1, 8, 32, 64, 128 (depending on group)
  local cachesize="$6"  # 0 or 2500000

  # ---------------------------
  # Map cache to DB name
  # ---------------------------
  local db=""
  case "$cache" in
    none)    db="pebble" ;;
    lru)     db="pebblelru" ;;
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
  # Label: Density = ASYNC, others = NOASYNC
  # ---------------------------
  local label="NOASYNC"
  if [ "$cache" = "density" ]; then
    label="ASYNC"
  fi

  # ---------------------------
  # Output file name
  # ---------------------------
  local outfile="${group}_wl${wl}_${cache}_${dist}_t${threads}_${label}.out"

  echo "Running ${group} wl${wl} cache=${cache} dist=${dist} threads=${threads} -> ${outfile}"

  # ---------------------------
  # Base go-ycsb run command
  # ---------------------------
  cmd="bin/go-ycsb run ${db} \
    -P ${wlfile} \
    --threads ${threads} \
    -p recordcount=${RECORDCOUNT} \
    -p operationcount=${OPCOUNT} \
    -p fieldcount=${FIELDCOUNT} \
    -p fieldlength=${FIELDLEN} \
    -p pebble.dir=${DBDIR} \
    -p dropdata=false \
    -p requestdistribution=${dist}"

  # Hotspot extra params (0.2, 0.8)
  if [ "${dist}" = "hotspot" ]; then
    cmd="${cmd} -p hotspotdatafraction=0.2 -p hotspotopnfraction=0.8"
  fi

  # ---------------------------
  # Set cache capacity for cached variants
  # ---------------------------
  if [ "${cachesize}" -gt 0 ]; then
    case "$cache" in
      lru)
        cmd="${cmd} -p pebble.lru_capacity=${cachesize}"
        ;;
      density)
        cmd="${cmd} -p pebble.density_capacity=${cachesize}"
        ;;
      detox)
        cmd="${cmd} -p pebble.detox_capacity=${cachesize}"
        ;;
    esac
  fi

  # Run and log to .out
  eval "${cmd} > ${outfile} 2>&1"
}

# ======================================
# 1) WorkloadVsDistribution experiments
# ======================================
for wl in A B C; do
  for dist in zipfian uniform hotspot sequential; do
    # No cache
    run_experiment "WorkloadVsDistribution" "$wl" "none"    "$dist" 32 0

    # Cached variants
    run_experiment "WorkloadVsDistribution" "$wl" "lru"     "$dist" 32 2500000
    run_experiment "WorkloadVsDistribution" "$wl" "density" "$dist" 32 2500000
    run_experiment "WorkloadVsDistribution" "$wl" "detox"   "$dist" 32 2500000
  done
done

# ======================================
# 2) MultithreadScaling experiments
# ======================================
for threads in 1 8 16 32 64 128; do
  # No cache
  run_experiment "MultithreadScaling" "A" "none"    "zipfian" "${threads}" 0

  # Cached variants
  run_experiment "MultithreadScaling" "A" "lru"     "zipfian" "${threads}" 2500000
  run_experiment "MultithreadScaling" "A" "density" "zipfian" "${threads}" 2500000
  run_experiment "MultithreadScaling" "A" "detox"   "zipfian" "${threads}" 2500000
done
