import json
from pathlib import Path

SOURCE = Path("cad-cardiac-mri-project-dvm.ipynb")

# Verificare
if not SOURCE.exists():
    raise FileNotFoundError(f"Nu găsesc: {SOURCE}")

original = SOURCE.read_text(encoding="utf-8")

print("\n=== ORIGINAL ===")
print(f"Size:      {len(original.encode('utf-8')) / 1024:.2f} KB")
print(f"Chars:     {len(original):,}")
print(f"Spaces:    {original.count(' '):,}")
print(f"Newlines:  {original.count(chr(10)):,}")
print(f"Tabs:      {original.count(chr(9)):,}")

nb = json.loads(original)

# --------------------------------------------------
# 1. Doar eliminare whitespace inutil din JSON
# --------------------------------------------------
compact = json.dumps(
    nb,
    ensure_ascii=False,
    separators=(",", ":")
)

P1 = SOURCE.with_name(SOURCE.stem + "_compact.ipynb")
P1.write_text(compact, encoding="utf-8")

# --------------------------------------------------
# 2. Eliminare OUTPUTS, dar păstrăm codul și markdown
# --------------------------------------------------
nb_no_outputs = json.loads(original)

for cell in nb_no_outputs.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

no_outputs = json.dumps(
    nb_no_outputs,
    ensure_ascii=False,
    separators=(",", ":")
)

P2 = SOURCE.with_name(SOURCE.stem + "_no_outputs.ipynb")
P2.write_text(no_outputs, encoding="utf-8")

# --------------------------------------------------
# 3. Eliminare OUTPUTS + metadata inutilă
# --------------------------------------------------
nb_clean = json.loads(original)

for cell in nb_clean.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

    # păstrăm metadata structurii, dar eliminăm metadata
    # de la celule dacă nu este necesară
    # (nu modificăm sursa codului)

# Compactare JSON
clean = json.dumps(
    nb_clean,
    ensure_ascii=False,
    separators=(",", ":")
)

P3 = SOURCE.with_name(SOURCE.stem + "_clean.ipynb")
P3.write_text(clean, encoding="utf-8")

# --------------------------------------------------
# REZULTATE
# --------------------------------------------------

def size_kb(path):
    return path.stat().st_size / 1024

print("\n=== COMPARAȚIE ===")
print(f"Original:     {size_kb(SOURCE):8.2f} KB")
print(f"Compact:      {size_kb(P1):8.2f} KB")
print(f"No outputs:   {size_kb(P2):8.2f} KB")
print(f"Clean:        {size_kb(P3):8.2f} KB")

print("\nFișiere create:")
print(P1)
print(P2)
print(P3)