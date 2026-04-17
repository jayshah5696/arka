def test_cli_handles_missing_config(tmp_path, monkeypatch, capsys):
    import pytest

    from arka.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--config", str(tmp_path / "non_existent.yaml")])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Configuration file not found" in captured.err

def test_cli_handles_invalid_config(tmp_path, monkeypatch, capsys):
    import pytest

    from arka.cli import main

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("invalid_yaml: {")

    with pytest.raises(SystemExit) as exc:
        main(["--config", str(config_path)])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Configuration is invalid" in captured.err or "while parsing a flow node" in captured.err
