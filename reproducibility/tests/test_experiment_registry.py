import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproducibility.run_experiment import (
    evaluation_commands,
    experiment_by_id,
    load_manifest,
    prepare_plan_aliases,
    question_experiments,
    training_command,
)


class ExperimentRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.defaults = cls.manifest["defaults"]

    def test_ids_are_unique(self):
        ids = [experiment["id"] for experiment in self.manifest["experiments"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_registered_blocks_match_cli_indices(self):
        # load_manifest performs the authoritative validation. This assertion
        # ensures that every channel-preserve run declares human-readable blocks.
        for experiment in self.manifest["experiments"]:
            if experiment["method"] == "channel_preserve" and not experiment.get(
                "full_body_freeze"
            ):
                self.assertTrue(experiment.get("blocks"), experiment["id"])

    def test_gradient_normalized_command_contains_calibration_flags(self):
        experiment = experiment_by_id(
            self.manifest, "ecpc_middle_gradient_normalized"
        )
        with patch.dict(os.environ, {"PAPER_INIT_CHECKPOINT": "/tmp/base"}):
            command = training_command(experiment, self.defaults)
        self.assertIn("--gradient_lr_control", command)
        self.assertEqual(command[command.index("--gradient_calibration_batches") + 1], "20")
        self.assertEqual(command[command.index("--gradient_lr_reference") + 1], "B")

    def test_evaluation_covers_each_task(self):
        experiment = experiment_by_id(self.manifest, "ecpc_e3_gradient_normalized")
        commands = evaluation_commands(
            experiment, self.defaults, self.manifest["datasets"]
        )
        evaluated = [command[command.index("-evaluate_on") + 1] for command in commands]
        self.assertEqual(evaluated, ["666", "555"])

    def test_unknown_experiment_fails_cleanly(self):
        with self.assertRaises(SystemExit):
            experiment_by_id(self.manifest, "does_not_exist")

    def test_each_question_resolves_to_unique_experiments(self):
        self.assertEqual(sorted(self.manifest["questions"]), ["Q1", "Q2", "Q3"])
        for question in self.manifest["questions"]:
            ids = [exp["id"] for exp in question_experiments(self.manifest, question)]
            self.assertTrue(ids, question)
            self.assertEqual(len(ids), len(set(ids)), question)

    def test_uniform_lr_control_compares_identical_region(self):
        experiments = question_experiments(self.manifest, "Q3")
        self.assertEqual({tuple(exp["blocks"]) for exp in experiments},
                         {("E4", "E5", "B", "D5", "D4")})

    def test_plan_aliases_copy_each_tasks_own_base_plans(self):
        experiment = experiment_by_id(self.manifest, "ecpc_e4_b_d4_uniform_lr0p005")
        datasets = self.manifest["datasets"]
        with tempfile.TemporaryDirectory() as root:
            bases = {}
            for task, content in (("666", b"umd plans"), ("555", b"ecpc plans")):
                task_dir = Path(root) / datasets[task]["name"]
                task_dir.mkdir()
                (task_dir / "nnUNetPlansv2.1_plans_3D.pkl").write_bytes(content)
                bases[task] = content
            with patch.dict(os.environ, {"nnUNet_preprocessed": root}):
                prepare_plan_aliases(experiment, datasets, execute=True)
                for task, content in bases.items():
                    alias = Path(root) / datasets[task]["name"] / f"{experiment['plan']}_plans_3D.pkl"
                    self.assertEqual(alias.read_bytes(), content)

                # An alias copied from another task must be refused.
                ecpc_alias = Path(root) / datasets["555"]["name"] / f"{experiment['plan']}_plans_3D.pkl"
                ecpc_alias.write_bytes(bases["666"])
                with self.assertRaises(SystemExit):
                    prepare_plan_aliases(experiment, datasets, execute=True)

    def test_dry_run_does_not_create_plan_aliases(self):
        experiment = experiment_by_id(self.manifest, "ecpc_e4_b_d4_uniform_lr0p0025")
        datasets = self.manifest["datasets"]
        with tempfile.TemporaryDirectory() as root:
            for task in ("666", "555"):
                task_dir = Path(root) / datasets[task]["name"]
                task_dir.mkdir()
                (task_dir / "nnUNetPlansv2.1_plans_3D.pkl").write_bytes(task.encode())
            with patch.dict(os.environ, {"nnUNet_preprocessed": root}):
                prepare_plan_aliases(experiment, datasets, execute=False)
            created = list(Path(root).rglob(f"{experiment['plan']}_plans_3D.pkl"))
            self.assertEqual(created, [])


if __name__ == "__main__":
    unittest.main()
