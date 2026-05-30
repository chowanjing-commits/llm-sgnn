"""
Verify repository hygiene before committing experiment work.

The project keeps large local artifacts and generated experiment outputs out of
Git. A small allowlist of curated paper-result CSV files under logs/ is tracked;
everything else in data/, embeddings/, models/, checkpoints/, and logs/ should
remain ignored or untracked.
"""
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TRACKED_DIRS = ("data/", "embeddings/", "models/", "checkpoints/")
FORBIDDEN_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".pkl",
    ".npz",
    ".npy",
}
MAX_TRACKED_FILE_BYTES = 1_000_000


def git_lines(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_log_allowlist():
    allowlist = set()
    for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("!logs/"):
            allowlist.add(line[1:].replace("\\", "/"))
    return allowlist


def file_size(path):
    return (PROJECT_ROOT / path).stat().st_size


def main():
    tracked_files = git_lines("ls-files")
    errors = []

    forbidden_dir_files = [
        path for path in tracked_files if path.startswith(FORBIDDEN_TRACKED_DIRS)
    ]
    if forbidden_dir_files:
        errors.append(
            "Tracked files found under forbidden artifact directories: "
            + ", ".join(forbidden_dir_files[:20])
        )

    forbidden_extension_files = [
        path for path in tracked_files if Path(path).suffix.lower() in FORBIDDEN_EXTENSIONS
    ]
    if forbidden_extension_files:
        errors.append(
            "Tracked generated/binary artifact files found: "
            + ", ".join(forbidden_extension_files[:20])
        )

    oversized = [
        (path, file_size(path))
        for path in tracked_files
        if (PROJECT_ROOT / path).is_file() and file_size(path) > MAX_TRACKED_FILE_BYTES
    ]
    if oversized:
        errors.append(
            "Tracked files exceed size limit: "
            + ", ".join(f"{path}={size}" for path, size in oversized[:20])
        )

    log_allowlist = load_log_allowlist()
    tracked_logs = [path for path in tracked_files if path.startswith("logs/")]
    unexpected_logs = [path for path in tracked_logs if path not in log_allowlist]
    if unexpected_logs:
        errors.append(
            "Tracked logs are not in .gitignore allowlist: "
            + ", ".join(unexpected_logs[:20])
        )

    if errors:
        raise AssertionError("\n".join(errors))

    print(
        "Repository hygiene verification passed: no tracked large artifacts, "
        "forbidden artifact directories, or unexpected log files."
    )


if __name__ == "__main__":
    main()
