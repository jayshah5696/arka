from __future__ import annotations

from pathlib import Path

from arka.pipeline.checkpoint import CheckpointManager


def test_checkpoint_manager_saves_and_loads_stage_artifact(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "state.db")
    stage_path = tmp_path / "runs" / "run-1" / "stages" / "01_source" / "data.parquet"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("placeholder")

    manager.register_run(run_id="run-1", config_hash="abc123", status="running")
    manager.save_stage(
        run_id="run-1",
        stage_name="01_source",
        artifact_path=stage_path,
        count_in=0,
        count_out=1,
        status="completed",
    )

    loaded_path = manager.load_stage(run_id="run-1", stage_name="01_source")
    runs = manager.list_runs()
    stage_runs = manager.list_stage_runs(run_id="run-1")

    assert loaded_path == stage_path
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["status"] == "running"
    assert stage_runs == [
        {
            "run_id": "run-1",
            "stage_name": "01_source",
            "artifact_path": str(stage_path),
            "count_in": 0,
            "count_out": 1,
            "status": "completed",
        }
    ]
