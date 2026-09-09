#!/usr/bin/env bash
# Qwen3-TTS single-GPU L0: CUDA MPS OFF vs ON A/B
#
# Zero-copy experiment: one Qwen3-TTS serve process (stage0 talker + stage1
# Code2Wav, both on the same GPU as two EngineCore subprocesses). We toggle
# only the CUDA MPS daemon and keep every other knob identical, then sweep
# client concurrency and measure qps / latency / SM utilization.
#
# Designed to run INSIDE the scheduler allocation on the remote omni box:
#   su - dyy -c 'source /data/dyy/env_setup.sh; dyy >/dev/null; \
#     cd /data/dyy/code/vllm-omni && \
#     gpu run --gpus 1 --timeout 40m --note "qwen3tts L0 mps A/B" -- \
#       bash mps_exp/qwen3tts_l0/run_l0.sh'
#
# Env knobs (all optional):
#   MODEL        default: first cached Qwen3-TTS CustomVoice, else the 1.7B id
#   PORT         server port (default 18091)
#   CONC_LIST    whitespace-separated client concurrency sweep (default "16 32 64")
#   N_MULT       prompts per concurrency level = concurrency * N_MULT (default 6)
#   WARMUP       warmup requests before the timed runs (default 6)
#   SRV_TIMEOUT  server health wait in seconds (default 900)
#
# NOTE: this script deliberately does NOT use `set -u`; env_setup.sh expands
# $PYTHONPATH without a `:-` guard, which would kill the script silently.

source /data/dyy/env_setup.sh
dyy >/dev/null 2>&1
[ -f /data/dyy/bench_ab/env_fix.sh ] && . /data/dyy/bench_ab/env_fix.sh

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/dyy_triton_cache_l0}"
mkdir -p "$TRITON_CACHE_DIR"

REPO_DIR=/data/dyy/code/vllm-omni
cd "$REPO_DIR" || { echo "[fatal] cannot cd $REPO_DIR"; exit 2; }
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

L0_DIR="$REPO_DIR/mps_exp/qwen3tts_l0"
OUT_DIR="$L0_DIR/out/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

PORT="${PORT:-18091}"
CONC_LIST="${CONC_LIST:-16 32 64}"
N_MULT="${N_MULT:-6}"
WARMUP="${WARMUP:-6}"
SRV_TIMEOUT="${SRV_TIMEOUT:-900}"
MPS_DIR="/tmp/dyy_mps_l0_$(id -u)"
BASE="http://127.0.0.1:${PORT}"
DEPLOY="vllm_omni/deploy/qwen3_tts.yaml"

echo "======================================================"
echo " L0 Qwen3-TTS MPS A/B   git=$GIT_HASH"
echo " CWD=$(pwd)"
echo " CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " PORT=$PORT CONC_LIST='$CONC_LIST' N_MULT=$N_MULT WARMUP=$WARMUP"
echo " OUT_DIR=$OUT_DIR"
echo "======================================================"

# ---- pick a model id: prefer one already in the HF cache -----------------
MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
    for cand in \
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" \
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; do
        cache="/data/models/hub/models--${cand//\//--}"
        if [ -d "$cache" ]; then MODEL="$cand"; break; fi
    done
fi
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
cache_dir="/data/models/hub/models--${MODEL//\//--}"
if [ -d "$cache_dir" ]; then
    export HF_HUB_OFFLINE=1
    echo "model $MODEL  (offline, HF cache hit)"
else
    export HF_HUB_OFFLINE=0
    echo "model $MODEL  (not in HF cache -> will try to download)"
fi
echo "cache_dir=$cache_dir"

# ---- scheduler must hand us exactly the GPUs to use -----------------------
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[fatal] CUDA_VISIBLE_DEVICES unset - run inside 'gpu run'"; exit 3
fi
PHYS_ID="${CUDA_VISIBLE_DEVICES%%,*}"
echo "physical gpu id for sampling: $PHYS_ID  (CVD='$CUDA_VISIBLE_DEVICES')"
nvidia-smi -L

# ---- helpers ---------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT_DIR/driver.log"; }

wait_health() {
    local deadline=$(( $(date +%s) + SRV_TIMEOUT ))
    while ! curl -fsS -o /dev/null "$BASE/health"; do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            log "server process died during startup"
            tail -n 60 "$SERVER_LOG"
            return 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            log "server health timeout after ${SRV_TIMEOUT}s"
            tail -n 80 "$SERVER_LOG"
            return 1
        fi
        sleep 5
    done
    return 0
}

start_server() {
    SERVER_LOG="$OUT_DIR/server_${MODE}.log"
    log "starting server (mode=$MODE) -> $SERVER_LOG"
    setsid nohup vllm serve "$MODEL" --omni \
        --deploy-config "$DEPLOY" \
        --host 127.0.0.1 --port "$PORT" \
        >"$SERVER_LOG" 2>&1 &
    SRV_PID=$!
    wait_health || return 1
    log "server healthy (pid=$SRV_PID)"
    nvidia-smi -i "$PHYS_ID" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader | tee -a "$OUT_DIR/gpu_after_${MODE}.txt"
    return 0
}

stop_server() {
    if [ -n "${SRV_PID:-}" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        log "stopping server group -${SRV_PID}"
        kill -TERM -- "-$SRV_PID" 2>/dev/null
        sleep 8
        kill -KILL -- "-$SRV_PID" 2>/dev/null
    fi
    SRV_PID=""
    sleep 2
}

start_mps() {
    rm -rf "$MPS_DIR"
    mkdir -p "$MPS_DIR" && chmod 700 "$MPS_DIR"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR"
    nvidia-cuda-mps-control -d
    sleep 4
    if [ -f "$MPS_DIR/nvidia-cuda-mps-control.pid" ]; then
        log "MPS daemon up (pid file present)"
    else
        log "WARNING: MPS daemon pid file not found after start"
    fi
}

stop_mps() {
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        echo quit | nvidia-cuda-mps-control 2>/dev/null
        sleep 3
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    fi
    pkill -f 'nvidia-cuda-mps-control' 2>/dev/null
    sleep 1
    log "MPS daemon stopped"
}

cleanup() {
    log "cleanup on exit"
    stop_server
    stop_mps
}
trap cleanup EXIT

# ---- run one mode ----------------------------------------------------------
run_mode() {
    MODE="$1"
    log "########## MODE=$MODE ##########"

    if [ "$MODE" = "on" ]; then
        start_mps || return 1
    else
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    fi

    start_server || return 1

    if [ "$MODE" = "on" ]; then
        log "MPS attach check:"
        python3 "$L0_DIR/mps_check.py" "$MPS_DIR" "$PORT" \
            | tee -a "$OUT_DIR/driver.log"
    fi

    # warmup a couple of requests so graphs/batching settle
    python3 "$L0_DIR/bench_client.py" \
        --base-url "$BASE" --model "$MODEL" \
        --concurrency 2 --num-prompts "$WARMUP" \
        --output-json "$OUT_DIR/warmup_${MODE}.json" \
        >>"$OUT_DIR/bench_${MODE}.log" 2>&1

    for C in $CONC_LIST; do
        N=$(( C * N_MULT ))
        log "bench mode=$MODE conc=$C n=$N"
        nvidia-smi dmon -i "$PHYS_ID" -s ucm -d 2 \
            >"$OUT_DIR/dmon_${MODE}_c${C}.csv" 2>&1 &
        DMON_PID=$!
        python3 "$L0_DIR/bench_client.py" \
            --base-url "$BASE" --model "$MODEL" \
            --concurrency "$C" --num-prompts "$N" \
            --output-json "$OUT_DIR/result_${MODE}_c${C}.json" \
            | tee -a "$OUT_DIR/bench_${MODE}.log"
        kill "$DMON_PID" 2>/dev/null
        wait "$DMON_PID" 2>/dev/null
        sleep 2
    done

    stop_server
    if [ "$MODE" = "on" ]; then
        stop_mps
    fi
}

run_mode off
run_mode on

# ---- summary --------------------------------------------------------------
echo "======================================================"
echo " SUMMARY (all runs done)"
python3 - "$OUT_DIR" "$PHYS_ID" <<'PY'
import csv, glob, json, os, statistics, sys

out_dir, phys = sys.argv[1], sys.argv[2]

def sm_avg(path: str) -> float:
    vals = []
    header = None
    try:
        with open(path) as f:
            for line in f:
                toks = line.split()
                if not toks:
                    continue
                if toks[0] == "#":
                    if header is None and "sm" in toks:
                        header = toks
                    continue
                if header is None:
                    continue
                try:
                    vals.append(float(toks[header.index("sm")]))
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        pass
    return round(statistics.fmean(vals), 1) if vals else float("nan")

rows = []
for mode in ("off", "on"):
    for jp in sorted(glob.glob(f"{out_dir}/result_{mode}_c*.json")):
        d = json.load(open(jp))
        c = d["concurrency"]
        rows.append((
            mode, c, d["ok"], d["errors"], d["qps"],
            d["latency_s"]["p50"], d["latency_s"]["p95"],
            d["latency_s"]["p99"], sm_avg(f"{out_dir}/dmon_{mode}_c{c}.csv"),
        ))

print(f"{'mode':<4} {'conc':>4} {'ok':>5} {'err':>4} {'qps':>8} {'p50':>6} {'p95':>6} {'p99':>6} {'sm%':>6}")
for mode, c, ok, err, qps, p50, p95, p99, sm in rows:
    print(f"{mode:<4} {c:>4} {ok:>5} {err:>4} {qps:>8.2f} {p50:>6.2f} {p95:>6.2f} {p99:>6.2f} {sm:>6.1f}")

print(f"\n(peak SM during each run; {phys} was sampled)")
PY

log "L0 done. artifacts under $OUT_DIR"
