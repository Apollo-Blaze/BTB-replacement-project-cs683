
#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Simulator configuration
BRANCH_PRED="hashed_perceptron"
L1D_PREF="fdip"
L2C_PREF="next_line"
LLC_PREF="spp_dev"
LLC_REPL="no"
L1D_REPL="lru"
CORES=1

WARMUP=50
SIM=50
MAX_JOBS=4

TRACE_DIR="$ROOT/dpc3_traces"
OUT="$ROOT/btb_experiment_results"
LOG_DIR="$OUT/logs"
MANIFEST="$OUT/run_manifest.csv"

mkdir -p "$LOG_DIR"
echo "model,trace,status,log" > "$MANIFEST"

MODELS=(
    convBTB
    convBTB_priority
    lfu
    lfu_priority
    lfu_priority_target
    lfu_priority_target_reuse_d
    lfu_target_priority
    lfu_priority_target_ind
    lru_target
    target_lfu_priority
    conditional_first
    call_first
    indirect_first
    return_first
)

TRACES=(
    spec_gcc_001
    spec_gobmk_001
    spec_perlbench_001
    spec_x264_001
    client_001
    client_005
    server_001
    server_010
)

# Track the current batch of parallel runs.
PIDS=()
BATCH_TRACES=()
BATCH_LOGS=()

wait_batch() {
    local i status

    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            status="SUCCESS"
            echo "Completed: ${CURRENT_MODEL} | ${BATCH_TRACES[$i]}"
        else
            status="FAILED"
            echo "FAILED: ${CURRENT_MODEL} | ${BATCH_TRACES[$i]} (see ${BATCH_LOGS[$i]})"
        fi

        echo "${CURRENT_MODEL},${BATCH_TRACES[$i]},${status},${BATCH_LOGS[$i]}" \
            >> "$MANIFEST"
    done

    PIDS=()
    BATCH_TRACES=()
    BATCH_LOGS=()
}

echo "ChampSim root: $ROOT"
echo "Warmup: $WARMUP"
echo "Simulation: $SIM"
echo "Parallel jobs: $MAX_JOBS"
echo "Models: ${#MODELS[@]}"
echo "Traces: ${#TRACES[@]}"
echo "Total runs: $((${#MODELS[@]} * ${#TRACES[@]}))"
echo

for model in "${MODELS[@]}"; do
    CURRENT_MODEL="$model"

    echo "========================================"
    echo "Building model: $model"
    echo "========================================"

    marker="$(mktemp)"
    candidates="$(mktemp)"
    touch "$marker"

    if ! ./build_champsim.sh \
        "$BRANCH_PRED" "$model" \
        "$L1D_PREF" "$L2C_PREF" "$LLC_PREF" \
        "$LLC_REPL" "$L1D_REPL" "$CORES"; then
        echo "BUILD FAILED: $model"
        echo "$model,ALL,BUILD_FAILED," >> "$MANIFEST"
        rm -f "$marker" "$candidates"
        continue
    fi

    # Find the executable produced or updated by this build.
    find "$ROOT/bin" -type f -perm -111 -newer "$marker" \
        -print > "$candidates"

    binary=""
    while IFS= read -r candidate; do
        if [[ -z "$binary" || "$candidate" -nt "$binary" ]]; then
            binary="$candidate"
        fi
    done < "$candidates"

    rm -f "$marker" "$candidates"

    if [[ -z "$binary" ]]; then
        echo "ERROR: Could not find executable for $model"
        echo "$model,ALL,BINARY_NOT_FOUND," >> "$MANIFEST"
        continue
    fi

    binary_name="$(basename "$binary")"
    binary_name="${binary_name%.*}"

    echo "Executable: $binary_name"
    echo

    for trace_name in "${TRACES[@]}"; do
        trace="$TRACE_DIR/${trace_name}.champsimtrace.xz"
        log="$LOG_DIR/${model}_${trace_name}.log"

        if [[ ! -f "$trace" ]]; then
            echo "MISSING TRACE: $trace"
            echo "$model,$trace_name,TRACE_MISSING,$log" >> "$MANIFEST"
            continue
        fi

        echo "Starting: $model | $trace_name"

        ./run_champsim.sh \
            "$binary_name" "$WARMUP" "$SIM" "${trace_name}.champsimtrace.xz" \
            > "$log" 2>&1 &

        PIDS+=("$!")
        BATCH_TRACES+=("$trace_name")
        BATCH_LOGS+=("$log")

        # Wait after launching three jobs before launching more.
        if [[ "${#PIDS[@]}" -ge "$MAX_JOBS" ]]; then
            wait_batch
        fi
    done

    # Wait for any remaining jobs (one or two traces).
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        wait_batch
    fi

    echo "Finished model: $model"
    echo
done

echo "========================================"
echo "All experiments finished."
echo "Results:  $OUT"
echo "Logs:     $LOG_DIR"
echo "Manifest: $MANIFEST"
echo "========================================"
