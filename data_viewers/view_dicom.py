# list_dicom_series.py
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def plane_from_iop(iop):
    if iop is None or len(iop) != 6:
        return "?"
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return ["sagittal", "coronal", "axial"][int(np.argmax(np.abs(n)))]


def first(v, default=None):
    return default if v in (None, "") else v

root = '/users/project1/pt01190/sjs/YSP_M_Kozeriew/MK001'

root = Path(root)
series = defaultdict(list)
skipped = 0

for f in root.rglob("*"):
    if not f.is_file():
        continue
    try:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
    except Exception:
        skipped += 1
        continue
    if "SeriesInstanceUID" not in ds:      # not an image DICOM (e.g. DICOMDIR, junk)
        skipped += 1
        continue
    series[str(ds.SeriesInstanceUID)].append((f, ds))

rows = []
for uid, items in series.items():
    ds0 = items[0][1]
    paths = [f for f, _ in items]
    positions = {tuple(np.round(np.array(d.ImagePositionPatient, float), 2))
                    for _, d in items if "ImagePositionPatient" in d}
    n_inst = len(items)
    frames = int(first(getattr(ds0, "NumberOfFrames", None), 1))
    n_slices = len(positions) if positions else n_inst
    n_vol = max(1, round(n_inst * frames / max(1, n_slices * frames))) if positions else 1

    px = [float(x) for x in first(getattr(ds0, "PixelSpacing", None), [np.nan, np.nan])]
    image_types = sorted({"\\".join(map(str, d.ImageType)) for _, d in items if "ImageType" in d})

    max_pixel_value = ds0.get("LargestImagePixelValue", None)

    rows.append(dict(
        series_no=int(first(getattr(ds0, "SeriesNumber", None), -1)),
        modality=getattr(ds0, "Modality", "?"),
        series_desc=getattr(ds0, "SeriesDescription", ""),
        protocol=getattr(ds0, "ProtocolName", ""),
        rows=getattr(ds0, "Rows", None),
        cols=getattr(ds0, "Columns", None),
        n_files=n_inst,
        n_slices=n_slices,
        n_volumes_or_contrasts=n_vol,
        multiframe=frames > 1,
        plane=plane_from_iop(getattr(ds0, "ImageOrientationPatient", None)),
        pixel_spacing=f"{px[0]:.3f}x{px[1]:.3f}",
        slice_thickness=getattr(ds0, "SliceThickness", None),
        spacing_between_slices=getattr(ds0, "SpacingBetweenSlices", None),
        TR=getattr(ds0, "RepetitionTime", None),
        TE=getattr(ds0, "EchoTime", None),
        image_types="; ".join(image_types)[:80],
        study_date=getattr(ds0, "StudyDate", ""),
        folder=str(paths[0].parent),
        series_uid=uid,
        max_pixel_value=max_pixel_value,
    ))

df = pd.DataFrame(rows).sort_values("series_no").reset_index(drop=True)
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 45)
print(df.drop(columns=["series_uid", "folder", "image_types"]).to_string())
print(f"\n{len(df)} series, {int(df.n_files.sum())} image files, {skipped} non-DICOM/skipped files")

# %%
# %%
for uid, items in series.items():
    ds0 = items[0][1]
    if ds0.SeriesDescription == 'ep2d_diff_b_0_10_20_30_50_80_100_300_400_500_800_1000_2000_ADC':
        print(ds0)
# %%