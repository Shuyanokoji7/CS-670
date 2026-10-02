from src.manifest import verify_manifest, write_manifest


def test_manifest_excludes_itself_and_keeps_superseded(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")
    assert write_manifest(tmp_path) == 2
    assert "MANIFEST" not in (tmp_path / "MANIFEST.sha256").read_text()
    assert verify_manifest(tmp_path) == []
    (tmp_path / "c.txt").write_text("z")
    assert write_manifest(tmp_path) == 3                       # superseded version retained, not overwritten
    assert len(list(tmp_path.glob("MANIFEST.sha256.superseded_*"))) == 1
    (tmp_path / "a.csv").write_text("changed")
    assert verify_manifest(tmp_path) == ["./a.csv"]
