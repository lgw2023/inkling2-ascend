"""CPU tests for launch policy; all deployment evidence and subprocesses are mocked.

These fixtures do not qualify a real Ascend installation or execute a model.
"""

import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_env


ROOT = Path(__file__).resolve().parents[1]


class RunnerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="runner-policy-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.model = self.work / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        self.qualification = self.work / "qualification.json"
        self.template = json.loads((ROOT / "configs/qualification.json").read_text(encoding="utf-8"))
        self.packages = {name: "cpu-test-fixture" for name in check_env.PACKAGES}
        self.data = copy.deepcopy(self.template)
        self.data["status"] = "qualified"
        self.data["locked_revisions"] = {key: "a" * 40 for key in self.data["locked_revisions"]}
        self.data["package_versions"] = self.packages.copy()
        self.data["runtime_versions"] = {key: "cpu-test-fixture" for key in self.data["runtime_versions"]}
        profile = self.data["profiles"]["a2"]
        profile["memory_total_bytes"] = 64 * 1024 ** 3
        profile["validated_tp_sizes"] = [2]
        profile["observed_runner_class"] = "vllm_ascend.worker.v1.model_runner.NPUModelRunner"
        profile["gates"] = {key: True for key in profile["gates"]}
        profile["evidence"] = ["CPU test fixture only: no actual device validation"]
        self.write_qualification(self.data)
        self.package_patch = patch.object(check_env, "package_versions", return_value=self.packages)
        self.package_patch.start()
        self.addCleanup(self.package_patch.stop)

    def write_qualification(self, data):
        self.qualification.write_text(json.dumps(data), encoding="utf-8")

    def args(self, execute=False):
        return argparse.Namespace(
            profile="a2", model=str(self.model), tp=2, devices="0,1",
            max_model_len=2048, port=8000, qualification=str(self.qualification),
            execute=execute,
        )

    def test_dry_run_disables_inherited_mrv2_without_loading_anything(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}), \
                patch.object(check_env, "qualification_errors") as qualification, \
                patch.object(check_env.subprocess, "run") as run, \
                patch.object(check_env.os, "execvpe") as execute, \
                contextlib.redirect_stdout(output):
            self.assertEqual(check_env.launch(self.args()), 0)
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "1")
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0 vllm serve", output.getvalue())
        self.assertNotIn("VLLM_USE_V1=0", output.getvalue())
        qualification.assert_not_called()
        run.assert_not_called()
        execute.assert_not_called()

    def test_cli_and_execute_share_v1_environment_despite_parent_mrv2_setting(self):
        args = self.args(execute=True)
        help_text = "\n".join(flag for flag in check_env.command_for(args) if flag.startswith("--"))
        help_result = subprocess.CompletedProcess([], 0, stdout=help_text, stderr="")
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}), \
                patch.object(check_env.shutil, "which", return_value="/test/vllm"), \
                patch.object(check_env.subprocess, "run", return_value=help_result) as run, \
                patch.object(check_env.os, "execvpe") as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_env.launch(args), 0)
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "1")
        run.assert_called_once()
        execute.assert_called_once()
        cli_env = run.call_args.kwargs["env"]
        exec_env = execute.call_args.args[2]
        self.assertEqual(cli_env, exec_env)
        self.assertEqual(exec_env["VLLM_USE_V2_MODEL_RUNNER"], "0")
        self.assertEqual(exec_env["ASCEND_RT_VISIBLE_DEVICES"], "0,1")

    def test_missing_or_v2_policy_is_rejected_even_with_other_evidence(self):
        policies = [None, {}, {"engine": "v0", "model_runner": "v1", "use_v2_model_runner": False},
                    {"engine": "v1", "model_runner": "v2", "use_v2_model_runner": False}]
        policies.extend({"engine": "v1", "model_runner": "v1", "use_v2_model_runner": value}
                        for value in (True, 0, "false", None))
        for policy in policies:
            with self.subTest(policy=policy):
                data = copy.deepcopy(self.data)
                if policy is None:
                    data.pop("runner_policy")
                else:
                    data["runner_policy"] = policy
                self.write_qualification(data)
                errors = check_env.qualification_errors(self.qualification, "a2", 2)
                self.assertTrue(any("runner_policy" in error for error in errors), errors)

    def test_recorded_runner_class_and_verified_gate_are_required(self):
        for field, value in (("observed_runner_class", None),
                             ("observed_runner_class", "vllm_ascend.worker.v2.model_runner.NPUModelRunner"),
                             ("v1_runner_verified", False)):
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.data)
                profile = data["profiles"]["a2"]
                if field == "v1_runner_verified":
                    profile["gates"][field] = value
                else:
                    profile[field] = value
                self.write_qualification(data)
                errors = check_env.qualification_errors(self.qualification, "a2", 2)
                self.assertTrue(any(field in error for error in errors), errors)

    def test_legacy_schema_cannot_skip_runner_evidence(self):
        data = copy.deepcopy(self.data)
        data["schema_version"] = 1
        self.write_qualification(data)
        self.assertTrue(any("schema_version" in error for error in
                            check_env.qualification_errors(self.qualification, "a2", 2)))

    def test_default_configuration_blocks_execute_for_every_profile(self):
        self.write_qualification(self.template)
        for profile in self.template["profiles"]:
            with self.subTest(profile=profile), \
                    patch.object(check_env.subprocess, "run") as run, \
                    patch.object(check_env.os, "execvpe") as execute, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                args = self.args(execute=True)
                args.profile = profile
                self.assertEqual(check_env.launch(args), 2)
                run.assert_not_called()
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
