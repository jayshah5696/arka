import json

from arka.cli import _print_summary


def test_print_summary_with_error(capsys, tmp_path):
    run_id = "test-error-run"
    report_dir = tmp_path / "runs" / run_id / "report"
    report_dir.mkdir(parents=True)

    report_data = {
        "status": "failed",
        "run_id": run_id,
        "final_count": 0,
        "stage_yields": [
            {
                "stage": "02_generate",
                "count_in": 10,
                "count_out": 5,
                "dropped_count": 0,
                "status": "failed",
                "error": {
                    "type": "LLMClientError",
                    "message": "API rate limit exceeded after 3 retries",
                },
            }
        ],
    }

    report_file = report_dir / "run_report.json"
    report_file.write_text(json.dumps(report_data))

    _print_summary(run_id, tmp_path)

    captured = capsys.readouterr()

    assert "02_generate: 10 in -> 5 out (lost 5 records) [failed]" in captured.out
    # stderr is used because the error lines are formatted with click.secho(..., err=True)
    assert (
        "Failed: LLMClientError: API rate limit exceeded after 3 retries"
        in captured.err
    )
