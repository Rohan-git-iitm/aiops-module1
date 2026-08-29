import csv
import pathlib

rows = sorted(str(p) for p in pathlib.Path("data").rglob("*") if p.is_file())
with open("filenames.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename"])
    w.writerows([r] for r in rows)
print(len(rows), "rows")
