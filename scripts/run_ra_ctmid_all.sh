#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="${RUN_GROUP:-all}"
FORCE="${FORCE:-0}"
DIAG_SPLIT="${DIAG_SPLIT:-valid}"
PYTHON="${PYTHON:-python}"

LOG_DIR="log/ra_ctmid_${RUN_ID}"
DIAG_DIR="${LOG_DIR}/diagnostics"
DRIVER_LOG="${LOG_DIR}/_driver.log"
SUMMARY="${LOG_DIR}/summary.tsv"
declare -A HANDLED_EXPERIMENTS=()

mkdir -p "${DIAG_DIR}"
touch "${DRIVER_LOG}"
if [[ ! -f "${SUMMARY}" ]]; then
  printf "timestamp\tname\tkind\tstatus\tcheckpoint\tlog\tbest_valid\n" > "${SUMMARY}"
fi

log_driver() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${DRIVER_LOG}"
}

append_summary() {
  local name="$1"
  local kind="$2"
  local status="$3"
  local checkpoint="$4"
  local log_file="$5"
  local best_valid="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "${name}" "${kind}" "${status}" "${checkpoint}" "${log_file}" "${best_valid}" >> "${SUMMARY}"
}

extract_best_valid() {
  local log_file="$1"
  if [[ -f "${log_file}" ]]; then
    awk '/best_valid=/{line=$0} END{sub(/^.*best_valid=/, "", line); print line}' "${log_file}"
  fi
}

checkpoint_for() {
  local name="$1"
  printf "checkpoints/ra_ctmid/%s/best.pt" "${name}"
}

base_config_for() {
  local domain="$1"
  case "${domain}" in
    toys) printf "configs/toys_and_games_ctmid.yaml" ;;
    home) printf "configs/home_and_kitchen_ctmid.yaml" ;;
    *) log_driver "ERROR unknown domain: ${domain}"; exit 2 ;;
  esac
}

run_experiment() {
  local domain="$1"
  local name="$2"
  local base_config
  local override_config
  local checkpoint
  local log_file
  local best_valid

  if [[ -n "${HANDLED_EXPERIMENTS[${name}]:-}" ]]; then
    log_driver "SKIP ${name}: already handled in this run"
    return
  fi

  base_config="$(base_config_for "${domain}")"
  override_config="configs/experiments/${name}.yaml"
  checkpoint="$(checkpoint_for "${name}")"
  log_file="${LOG_DIR}/${name}.log"

  if [[ ! -f "${override_config}" ]]; then
    log_driver "FAILED ${name}: missing config ${override_config}"
    append_summary "${name}" "train" "failed_missing_config" "${checkpoint}" "${log_file}" ""
    exit 1
  fi

  if [[ "${FORCE}" != "1" && -f "${checkpoint}" ]]; then
    best_valid="$(extract_best_valid "${log_file}")"
    log_driver "SKIP ${name}: checkpoint exists at ${checkpoint}"
    append_summary "${name}" "train" "skipped" "${checkpoint}" "${log_file}" "${best_valid:-checkpoint_exists}"
    HANDLED_EXPERIMENTS["${name}"]=1
    return
  fi

  log_driver "START ${name}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" run_ctmid.py \
    --config "${base_config}" \
    --config "${override_config}" 2>&1 | tee "${log_file}"; then
    best_valid="$(extract_best_valid "${log_file}")"
    log_driver "DONE ${name}"
    append_summary "${name}" "train" "done" "${checkpoint}" "${log_file}" "${best_valid:-missing_best_valid}"
    HANDLED_EXPERIMENTS["${name}"]=1
  else
    log_driver "FAILED ${name}"
    append_summary "${name}" "train" "failed" "${checkpoint}" "${log_file}" ""
    exit 1
  fi
}

run_diagnostic() {
  local domain="$1"
  local name="$2"
  local base_config
  local override_config
  local checkpoint
  local output_file
  local log_file
  base_config="$(base_config_for "${domain}")"
  override_config="configs/experiments/${name}.yaml"
  checkpoint="$(checkpoint_for "${name}")"
  output_file="${DIAG_DIR}/${name}_${DIAG_SPLIT}.json"
  log_file="${LOG_DIR}/diagnostics_${name}_${DIAG_SPLIT}.log"

  if [[ ! -f "${override_config}" ]]; then
    log_driver "FAILED diagnostics ${name}: missing config ${override_config}"
    append_summary "${name}" "diagnostics" "failed_missing_config" "${checkpoint}" "${log_file}" ""
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    log_driver "FAILED diagnostics ${name}: missing checkpoint ${checkpoint}"
    append_summary "${name}" "diagnostics" "failed_missing_checkpoint" "${checkpoint}" "${log_file}" ""
    exit 1
  fi
  if [[ "${FORCE}" != "1" && -f "${output_file}" ]]; then
    log_driver "SKIP diagnostics ${name}: output exists at ${output_file}"
    append_summary "${name}" "diagnostics" "skipped" "${checkpoint}" "${output_file}" "diagnostics_exists"
    return
  fi

  log_driver "START diagnostics ${name} split=${DIAG_SPLIT}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/evaluate_ra_ctmid_diagnostics.py \
    --config "${base_config}" \
    --config "${override_config}" \
    --split "${DIAG_SPLIT}" \
    --output "${output_file}" 2>&1 | tee "${log_file}"; then
    log_driver "DONE diagnostics ${name}"
    append_summary "${name}" "diagnostics" "done" "${checkpoint}" "${output_file}" "ok"
  else
    log_driver "FAILED diagnostics ${name}"
    append_summary "${name}" "diagnostics" "failed" "${checkpoint}" "${log_file}" ""
    exit 1
  fi
}

run_toys_residual() {
  local experiments=(
    toys_6modal_nores
    toys_6modal_last0p25
    toys_6modal_last0p5
    toys_6modal_idres0p25
    toys_6modal_idres0p5
    toys_6modal_idres0p75
    toys_6modal_idres1p0
    toys_6modal_idres0p5_last0p1
    toys_6modal_idres0p5_last0p25
    toys_6modal_idres0p5_last0p5
    toys_6modal_idres0p5_last0p75
    toys_6modal_idres0p5_last1p0
    toys_6modal_idres0p25_last0p5
    toys_6modal_idres0p75_last0p5
    toys_6modal_idres1p0_last0p5
    toys_6modal_idres0p5_rank0p1
    toys_6modal_idres0p5_rank0p3
  )
  local name
  for name in "${experiments[@]}"; do
    run_experiment toys "${name}"
  done
}

run_toys_backbone() {
  local experiments=(
    toys_6modal_idres0p5_last0p5
    toys_ra_without_ms_ode
    toys_ra_without_user_ode
    toys_ra_without_item_ode
    toys_ra_without_mgt
    toys_ra_without_taf
    toys_ra_exp_decay
  )
  local name
  for name in "${experiments[@]}"; do
    run_experiment toys "${name}"
  done
}

run_home() {
  local experiments=(
    home_5modal_nores
    home_6modal_nores
    home_6modal_last0p5
    home_6modal_idres0p5
    home_6modal_idres0p5_last0p25
    home_6modal_idres0p5_last0p5
  )
  local name
  for name in "${experiments[@]}"; do
    run_experiment home "${name}"
  done
}

run_diagnostics() {
  local toys_experiments=(
    toys_6modal_nores
    toys_6modal_last0p5
    toys_6modal_idres0p5
    toys_6modal_idres0p5_last0p5
  )
  local home_experiments=(
    home_6modal_nores
    home_6modal_idres0p5
    home_6modal_idres0p5_last0p5
  )
  local name
  for name in "${toys_experiments[@]}"; do
    run_diagnostic toys "${name}"
  done
  for name in "${home_experiments[@]}"; do
    run_diagnostic home "${name}"
  done
}

log_driver "RUN_ID=${RUN_ID} RUN_GROUP=${RUN_GROUP} GPU_ID=${GPU_ID} FORCE=${FORCE} DIAG_SPLIT=${DIAG_SPLIT}"

case "${RUN_GROUP}" in
  all)
    run_toys_residual
    run_toys_backbone
    run_home
    run_diagnostics
    ;;
  toys_residual)
    run_toys_residual
    ;;
  toys_backbone)
    run_toys_backbone
    ;;
  home)
    run_home
    ;;
  diagnostics)
    run_diagnostics
    ;;
  *)
    log_driver "ERROR unknown RUN_GROUP=${RUN_GROUP}; expected all, toys_residual, toys_backbone, home, diagnostics"
    exit 2
    ;;
esac

log_driver "COMPLETE RUN_GROUP=${RUN_GROUP}"
