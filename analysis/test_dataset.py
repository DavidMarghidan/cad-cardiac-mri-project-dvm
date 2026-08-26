from pathlib import Path
from collections import Counter
from pathlib import Path
import re

from pathlib import Path
from collections import defaultdict
import hashlib


ROOT = Path(r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset')

def file_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def get_series(path):
    # path:
    # Sick/Directory_24/SR_103/IM00001.jpg

    rel = path.relative_to(ROOT)

    # class / directory / series
    return tuple(rel.parts[:3])


# ---------------------------------------------------------
# 1. Hash all images
# ---------------------------------------------------------

hashes = defaultdict(list)

for p in ROOT.rglob("*.jpg"):
    hashes[file_hash(p)].append(p)


duplicates = {
    h: paths
    for h, paths in hashes.items()
    if len(paths) > 1
}


# ---------------------------------------------------------
# 2. Build graph between series
# ---------------------------------------------------------

graph = defaultdict(set)

cross_directory = []
cross_class = []

for paths in duplicates.values():

    series = sorted(set(get_series(p) for p in paths))

    for i in range(len(series)):
        for j in range(i + 1, len(series)):

            a = series[i]
            b = series[j]

            graph[a].add(b)
            graph[b].add(a)

            # a = (class, Directory, series)
            if a[0] != b[0]:
                cross_class.append((a, b))

            if a[1] != b[1]:
                cross_directory.append((a, b))


print("Series connected by duplicates:", len(graph))

print(
    "Cross-directory duplicate connections:",
    len(set(cross_directory))
)

print(
    "Cross-class duplicate connections:",
    len(set(cross_class))
)


# ---------------------------------------------------------
# 3. Connected components
# ---------------------------------------------------------

visited = set()
components = []

for node in graph:

    if node in visited:
        continue

    stack = [node]
    component = []

    while stack:

        x = stack.pop()

        if x in visited:
            continue

        visited.add(x)
        component.append(x)

        for y in graph[x]:
            if y not in visited:
                stack.append(y)

    components.append(component)


components.sort(key=len, reverse=True)

print("\nConnected components:", len(components))


for i, component in enumerate(components[:100], 1):

    print(
        f"\nCOMPONENT {i}: "
        f"{len(component)} series"
    )

    for cls, directory, series in sorted(component):
        print(
            f"  {cls}\\{directory}\\{series}"
        )