import os

from collections import defaultdict
from pathlib import Path

# Download latest version
path = Path(r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset')

print("=" * 80)
print("DATASET PATH")
print(path)
print("=" * 80)

# ---------------------------------------------------------
# 1. Show top-level structure
# ---------------------------------------------------------

print("\nTOP LEVEL:")
for item in sorted(os.listdir(path)):
    full = os.path.join(path, item)
    print(
        f"{'[DIR] ' if os.path.isdir(full) else '[FILE]'}{item}"
    )

# ---------------------------------------------------------
# 2. Find Normal / Sick
# ---------------------------------------------------------

normal_dir = None
sick_dir = None

for root, dirs, files in os.walk(path):
    for d in dirs:
        dl = d.lower()

        if dl == "normal":
            normal_dir = os.path.join(root, d)

        elif dl in ("sick", "abnormal", "cad"):
            sick_dir = os.path.join(root, d)

print("\nNORMAL:", normal_dir)
print("SICK:  ", sick_dir)

# ---------------------------------------------------------
# 3. Function to analyze one class
# ---------------------------------------------------------

def analyze_class(class_dir, class_name):

    if class_dir is None:
        print(f"\n{class_name}: NOT FOUND")
        return

    patients = []
    total_series = 0
    total_images = 0

    print(f"\n{'=' * 80}")
    print(f"{class_name}")
    print(f"{'=' * 80}")

    # immediate directories = candidate patients
    for patient_name in sorted(os.listdir(class_dir)):

        patient_path = os.path.join(class_dir, patient_name)

        if not os.path.isdir(patient_path):
            continue

        patients.append(patient_name)

        series_count = 0
        image_count = 0

        # Search inside patient
        for root, dirs, files in os.walk(patient_path):

            # Count image files
            image_files = [
                f for f in files
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm")
                )
            ]

            image_count += len(image_files)

        # Candidate series = directories directly below patient
        series_count = sum(
            os.path.isdir(os.path.join(patient_path, x))
            for x in os.listdir(patient_path)
        )

        total_series += series_count
        total_images += image_count

        print(
            f"{patient_name:25s}"
            f" series={series_count:4d}"
            f" images={image_count:6d}"
        )

    print("\nSUMMARY")
    print("-" * 80)
    print("Patients :", len(patients))
    print("Series   :", total_series)
    print("Images   :", total_images)

    if patients:
        print(
            "Avg series/patient:",
            round(total_series / len(patients), 2)
        )
        print(
            "Avg images/patient:",
            round(total_images / len(patients), 2)
        )

    return {
        "patients": patients,
        "patient_count": len(patients),
        "series_count": total_series,
        "image_count": total_images,
    }


normal = analyze_class(normal_dir, "NORMAL")
sick = analyze_class(sick_dir, "SICK")

# ---------------------------------------------------------
# 4. Final audit
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL DATASET AUDIT")
print("=" * 80)

if normal and sick:

    print("Normal patients :", normal["patient_count"])
    print("Sick patients   :", sick["patient_count"])
    print("TOTAL patients  :",
          normal["patient_count"] + sick["patient_count"])

    print()

    print("Normal series   :", normal["series_count"])
    print("Sick series     :", sick["series_count"])
    print("TOTAL series    :",
          normal["series_count"] + sick["series_count"])

    print()

    print("Normal images   :", normal["image_count"])
    print("Sick images     :", sick["image_count"])
    print("TOTAL images    :",
          normal["image_count"] + sick["image_count"])

print("=" * 80)