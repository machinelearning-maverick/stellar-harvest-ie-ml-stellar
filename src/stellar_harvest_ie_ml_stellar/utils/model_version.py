from pathlib import Path


def read_version(path: Path) -> str:
    version_file = path / "VERSION"
    return version_file.read_text().strip()
