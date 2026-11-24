#!/usr/bin/env python3
"""Read-only environment summary and guarded launch templates; stdlib only.

This is project tooling, not the Inkling2 NPU model implementation.
No package, driver, firmware or account changes are performed.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend", "fla-npu")
V1_RUNNER_CLASS = "vllm_ascend.worker.v1.model_runner.NPUModelRunner"
RUNNER_POLICY = {"engine": "v1", "model_runner": "v1", "use_v2_model_runner": False}
GATES = (
    "hardware_and_stack_verified", "v1_runner_verified", "model_adapter_registered", "bf16_loader_verified",
    "operator_parity_passed", "cache_isolation_passed", "tp_parity_passed", "text_e2e_passed",
)


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def collect(args):
    report = {
        "schema_version": 1,
        "python": platform.python_version(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "packages": package_versions(),
        "runtime_versions": {
            "driver": None, "firmware": None, "cann_toolkit": None,
            "cann_ops": None, "nnal": None,
        },
        "npu": {"probe_requested": args.probe_npu, "device_count": None, "devices": []},
        "note": "Only allowlisted fields. Fill runtime versions from the installed release metadata; no raw system logs are included.",
    }
    # Optional isolated probe: do not forward import warnings, raw errors or logs.
    if args.probe_npu:
        probe = """
import json, torch, torch_npu
rows = []
count = torch.npu.device_count()
for i in range(count):
    prop = torch.npu.get_device_properties(i)
    rows.append({'index': i, 'name': torch.npu.get_device_name(i),
                 'memory_total_bytes': getattr(prop, 'total_memory', None)})
print('INKLING_ENV_JSON=' + json.dumps({'device_count': count, 'devices': rows}))
"""
        try:
            done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=45)
            lines = [x for x in done.stdout.splitlines() if x.startswith("INKLING_ENV_JSON=")]
            if done.returncode or not lines:
                report["npu"]["probe_status"] = "unavailable"
            else:
                data = json.loads(lines[-1].split("=", 1)[1])
                report["npu"]["device_count"] = data["device_count"]
                report["npu"]["devices"] = data["devices"]
                report["npu"]["probe_status"] = "ok"
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
            report["npu"]["probe_status"] = "unavailable"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def qualification_errors(path, profile, tp):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["qualification file is missing or invalid JSON"]
    errors = []
    if data.get("schema_version") != 2:
        errors.append("qualification schema_version must be 2 with V1 runner evidence")
    if data.get("status") != "qualified":
        errors.append("qualification status must be qualified (current template is not qualified)")
    policy = data.get("runner_policy")
    if not isinstance(policy, dict):
        errors.append("runner_policy is required: V1 engine and Model Runner V1 only")
    else:
        for key, expected in RUNNER_POLICY.items():
            actual = policy.get(key)
            matches = (actual is False) if expected is False else (actual == expected)
            if not matches:
                errors.append(f"runner_policy.{key} must be {expected!r}; MRV2 is not supported")
    if data.get("weight_format") != "dequantized_bf16":
        errors.append("this launch template requires the separately verified BF16 conversion/loader path")
    for key in ("model", "vllm", "vllm_ascend", "adapter"):
        sha = data.get("locked_revisions", {}).get(key)
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            errors.append(f"locked_revisions.{key} must be a full commit hash")
    installed = package_versions()
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        expected = data.get("package_versions", {}).get(name)
        if not expected or expected != installed[name]:
            errors.append(f"package version is not locked to the installed {name}")
    for key in ("driver", "firmware", "cann_toolkit", "cann_ops", "nnal"):
        if not data.get("runtime_versions", {}).get(key):
            errors.append(f"runtime_versions.{key} is not recorded")
    selected = data.get("profiles", {}).get(profile, {})
    if selected.get("observed_runner_class") != V1_RUNNER_CLASS:
        errors.append(f"{profile}.observed_runner_class must record the verified {V1_RUNNER_CLASS}")
    if not isinstance(selected.get("memory_total_bytes"), int) or selected["memory_total_bytes"] <= 0:
        errors.append("measured per-device memory_total_bytes is required")
    if tp not in selected.get("validated_tp_sizes", []):
        errors.append(f"TP={tp} has no recorded qualification for {profile}")
    for gate in GATES:
        if selected.get("gates", {}).get(gate) is not True:
            errors.append(f"{profile}.{gate} has not passed")
    if not selected.get("evidence"):
        errors.append("qualification evidence paths or URLs are missing")
    return errors


def command_for(args):
    return [
        "vllm", "serve", args.model,
        "--served-model-name", "inkling2-mini",
        "--tensor-parallel-size", str(args.tp),
        "--pipeline-parallel-size", "1",
        "--distributed-executor-backend", "mp",
        "--dtype", "bfloat16", "--enforce-eager",
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_model_len),
        "--max-num-seqs", "1", "--gpu-memory-utilization", "0.80",
        "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
        "--limit-mm-per-prompt", '{"image":0}',
        "--seed", "0", "--host", "127.0.0.1", "--port", str(args.port),
    ]


def launch_environment(devices):
    """Apply the project runner choice to child processes, without changing the shell.

    This flag requires a qualified V1-capable branch. It does not restore V1 in
    upstream versions that removed it, nor prove which class was instantiated.
    """
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    return env


def validate_cli(command, env):
    if shutil.which("vllm") is None:
        return ["vllm is not installed in the selected environment"]
    try:
        done = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True, timeout=45, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return ["could not read the installed vLLM serve help"]
    if done.returncode:
        return ["installed vLLM does not return serve --help=all successfully; inspect its CLI manually"]
    help_text = re.sub(r"\x1b\[[0-9;]*m", "", done.stdout + done.stderr)
    return [f"installed vLLM CLI does not expose {flag}" for flag in command
            if flag.startswith("--") and not re.search(r"(?<![\w-])" + re.escape(flag) + r"(?=[\s,=]|$)", help_text)]


def launch(args):
    devices = args.devices or ",".join(map(str, range(args.tp)))
    if not re.fullmatch(r"\d+(,\d+)*", devices):
        raise ValueError("--devices must be comma-separated local numeric NPU IDs")
    ids = devices.split(",")
    if len(ids) != args.tp or len(set(ids)) != args.tp:
        raise ValueError("device count must equal TP and device IDs must be distinct")
    command = command_for(args)
    env = launch_environment(devices)
    print("Profile:", args.profile)
    print("Mode:", "EXECUTE after qualification checks" if args.execute else "DRY RUN (no model or device is loaded)")
    print("Runner: vLLM V1 engine / Ascend Model Runner V1 (MRV2 disabled)")
    print("ASCEND_RT_VISIBLE_DEVICES=" + devices + " VLLM_USE_V2_MODEL_RUNNER=0 " + shlex.join(command), flush=True)
    if not args.execute:
        print("This repository does not contain a completed Inkling2 Ascend adapter. See docs/deployment.md.")
        return 0
    errors = qualification_errors(args.qualification, args.profile, args.tp)
    if not (Path(args.model).is_dir() and (Path(args.model) / "config.json").is_file()):
        errors.append("--model must be a local qualified checkpoint directory with config.json")
    # Read the installed CLI only after recorded model qualification passes.
    if not errors:
        errors.extend(validate_cli(command, env))
    if errors:
        for error in errors:
            print("BLOCKED:", error, file=sys.stderr)
        return 2
    os.execvpe(command[0], command, env)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    collect_parser = actions.add_parser("collect", help="print allowlisted JSON; no device probe by default")
    collect_parser.add_argument("--probe-npu", action="store_true", help="read device count, model and total memory through torch_npu")
    run_parser = actions.add_parser("launch", help="review launch command, or execute a qualified local adapter")
    run_parser.add_argument("--profile", choices=("a2", "a5-950pr", "a5-950dt"), required=True)
    run_parser.add_argument("--model", default="./models/inkling2-mini-bf16")
    run_parser.add_argument("--tp", type=int, choices=(2, 4, 8), default=2)
    run_parser.add_argument("--devices", help="numeric local device IDs; defaults to the first TP devices")
    run_parser.add_argument("--max-model-len", type=int, choices=(512, 1024, 2048, 4096), default=2048)
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--qualification", default=str(ROOT / "configs" / "qualification.json"))
    run_parser.add_argument("--execute", action="store_true", help="requires all qualification gates and installed CLI validation")
    args = parser.parse_args()
    if args.action == "launch" and not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    try:
        return collect(args) if args.action == "collect" else launch(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
