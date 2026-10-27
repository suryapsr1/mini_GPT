import pathlib
import sys

# Standard library path of the active Python environment
stdlib_path = pathlib.Path(sys.base_prefix) / "Lib"

out_path = pathlib.Path("data/input.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

collected_bytes = 0
target_bytes = 150_000  # ~150 KB of clean code

with open(out_path, "w", encoding="utf-8") as outfile:
    for py_file in stdlib_path.glob("*.py"):
        # Skip internal, test, or setup files
        if py_file.name.startswith("_") or "test" in py_file.name.lower():
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            outfile.write(f"\n# File: {py_file.name}\n")
            outfile.write(content)
            collected_bytes += len(content.encode("utf-8"))

            if collected_bytes >= target_bytes:
                break
        except Exception:
            continue

print(
    f"Successfully compiled {collected_bytes / 1024:.2f} KB of Python code into {out_path}"
)