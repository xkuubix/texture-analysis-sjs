
import re
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import csv
import nibabel as nib
import nilearn.image as nimg


import shutil
warnings.filterwarnings("ignore", message=".*pixdim.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*slope.*", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# File-discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

# Maps logical sequence names → regex patterns that match the raw .nii.gz files
SEQUENCE_PATTERNS: Dict[str, str] = {
    "T1_VIBE_IN":    r"t1_vibe_dixon.*_in",         # in-phase
    "T1_VIBE_F":     r"t1_vibe_dixon.*_F(?!$)",     # fat-only
    "T1_VIBE_W":     r"t1_vibe_dixon.*_W(?!$)",     # water-only
    "T2_SPACE":      r"t2_space_tra.*iso_1\.0mm",
    "Sialo_MR":      r"Sialo-MR",
    "T1_TSE":        r"t1_tse_(?:tra|cor)",
    "T2_DIXON_IN":   r"t2_tse_dixon.*_in",          # in-phase
    "T2_DIXON_F":    r"t2_tse_dixon.*_F(?!$)",      # fat-only
    "T2_DIXON_W":    r"t2_tse_dixon.*_W(?!$)",      # water-only
    "ADC":           r"ep2d_diff",
}

# Maps logical segmentation names → file prefix and extraction mode.
SEG_PREFIXES: Dict[str, Dict[str, str]] = {
    "T1_VIBE_IN":  {"prefix": "T1_VIBE_in_Segmentation",    "mode": "3D"},
    "T1_VIBE_F":   {"prefix": "T1_VIBE_F_Segmentation",     "mode": "3D"},
    "T1_VIBE_W":   {"prefix": "T1_VIBE_W_Segmentation",     "mode": "3D"},
    "T2_SPACE":    {"prefix": "T2_SPACE_Segmentation",      "mode": "3D"},
    "Sialo_MR":    {"prefix": "Sialo_MR_Segmentation",      "mode": "3D"},
    "T1_TSE":      {"prefix": "T1_TSE_Segmentation",        "mode": "2D"},
    "T2_DIXON_F":  {"prefix": "T2_DIXON_F_Segmentation",    "mode": "2D"},
    "T2_DIXON_IN": {"prefix": "T2_DIXON_in_Segmentation",   "mode": "2D"},
    "T2_DIXON_W":  {"prefix": "T2_DIXON_W_Segmentation",    "mode": "2D"},
    "ADC":         {"prefix": "ADC_Segmentation",           "mode": "2D"},
}


def _find_file(folder: Path, pattern: str) -> Optional[Path]:
    """Return the first file in folder whose name matches regex pattern."""
    matches = [
        f
        for f in folder.iterdir()
        if f.is_file()
        and "-label.nii.gz" not in f.name
        and re.search(pattern, f.name, re.IGNORECASE)
    ]
    if len(matches) > 1:
        print(f"Multiple files in {folder} match pattern {pattern!r}; "
            f" {matches}.",
        )
    return matches[0] if matches else None


def _load_color_table(csv_path: Path) -> Dict[int, Tuple[str, Tuple[float, float, float]]]:
    """
    Parse a 3D Slicer colour-table CSV.
    Returns {label_index: (name, (r, g, b))} with colours in [0, 1].

    Slicer exports two formats:
      • 4-column: index, name, r, g, b, a  (values 0-255)
      • With header row starting with #
    """
    result: Dict[int, Tuple[str, Tuple[float, float, float]]] = {}
    if not csv_path.exists():
        return result

    with open(csv_path, newline="", encoding="utf-8") as fh:
        lines = [l for l in fh if not l.startswith("#") and l.strip()]

    reader = csv.reader(lines)
    for row in reader:
        row = [c.strip() for c in row]
        try:
            # Try standard Slicer format: index, name, r255, g255, b255 [, a255]
            if len(row) >= 5 and row[0].lstrip("-").isdigit():
                idx = int(row[0])
                name = row[1]
                r, g, b = int(row[2]) / 255.0, int(row[3]) / 255.0, int(row[4]) / 255.0
                result[idx] = (name, (r, g, b))
        except (ValueError, IndexError):
            continue
    return result


def _medical_geometry(image) -> Tuple[Tuple[int, int, int], str, str]:
    """Return shape in LR/AP/SI order, axis labels, and acquisition plane."""
    shape = tuple(int(value) for value in image.shape[:3])
    axcodes = nib.aff2axcodes(image.affine)
    axis_groups = {
        "L": "LR", "R": "LR",
        "A": "AP", "P": "AP",
        "S": "SI", "I": "SI",
    }
    group_indices = {axis_groups[code]: index for index, code in enumerate(axcodes)}
    medical_shape = tuple(shape[group_indices[group]] for group in ("LR", "AP", "SI"))
    axis_labels = " × ".join(f"{code} ({group})" for code, group in zip(axcodes, (axis_groups[code] for code in axcodes)))
    slice_axis = min(range(3), key=lambda index: shape[index])
    plane = {"LR": "sagittal", "AP": "coronal", "SI": "axial"}[axis_groups[axcodes[slice_axis]]]
    return medical_shape, axis_labels, plane


def _medical_voxel_sizes(image) -> Tuple[float, float, float]:
    """Return voxel sizes in the same LR/AP/SI order as _medical_geometry."""
    axcodes = nib.aff2axcodes(image.affine)
    axis_groups = {
        "L": "LR", "R": "LR",
        "A": "AP", "P": "AP",
        "S": "SI", "I": "SI",
    }
    group_indices = {axis_groups[code]: index for index, code in enumerate(axcodes)}
    spacing = image.header.get_zooms()[:3]
    return tuple(float(spacing[group_indices[group]]) for group in ("LR", "AP", "SI"))


def _data_metadata(image) -> str:
    """Return on-disk dtype, allocated bits, and finite data range."""
    data = np.asanyarray(image.dataobj)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        value_range = "empty"
    else:
        value_range = f"{float(finite.min()):.4g}..{float(finite.max()):.4g}"
    return f"{image.get_data_dtype()} / {int(image.header['bitpix'])}b / {value_range}"

GEOMETRY_TOLERANCE = 1e-3
STRICT_GEOMETRY_TOLERANCE = 1e-16


def _recommended_geometry_tolerance(image, mask) -> str:
    """Recommend a PyRadiomics tolerance from observed image/mask drift."""
    spacing_difference = np.max(np.abs(
        np.asarray(image.header.get_zooms()[:3])
        - np.asarray(mask.header.get_zooms()[:3])
    ))
    affine_difference = np.max(np.abs(image.affine - mask.affine))
    maximum_difference = float(max(spacing_difference, affine_difference))

    if maximum_difference <= STRICT_GEOMETRY_TOLERANCE:
        return "1e-16 (strict; exact match)"
    if maximum_difference <= GEOMETRY_TOLERANCE:
        exponent = int(np.ceil(np.log10(maximum_difference)))
        recommended = 10.0 ** exponent
        return f"{recommended:.0e} (sufficient for observed drift {maximum_difference:.3g})"
    return (
        f">1e-3 (observed {maximum_difference:.3g}; correct geometry first)"
    )


class Patient:
    """
    Represents one patient folder (e.g. MK080_Segmentacja).

    Attributes
    ----------
    folder : Path
    patient_id : str          e.g. "MK080"
    sequences : dict          {seq_name: nib.Nifti1Image or None}
    segmentations : dict      {seg_name: {"label": nib.Nifti1Image, "colors": dict}}
    """

    def __init__(self, folder: Union[str, Path]):
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Patient folder not found: {self.folder}")

        # Extract patient ID from folder name  (MK080_Segmentacja → MK080)
        self.id = re.match(r"(MK\d+(?:_[A-Z])?)(?:_|$)", self.folder.name)
        self.id = self.id.group(1) if self.id else self.folder.name

        self.sequences: Dict[str, Optional[Path]] = {}
        self.segmentations: Dict[str, Dict] = {}

        self._discover()

    # ── Discovery ─────────────────────────────────────────────────────────────
    def _discover(self):
        """Scan folder and populate sequences + segmentations dicts."""
        for seq_name, pattern in SEQUENCE_PATTERNS.items():
            self.sequences[seq_name] = _find_file(self.folder, pattern + r".*\.nii\.gz$")

        for seg_name, settings in SEG_PREFIXES.items():
            prefix = settings["prefix"]
            label_path = self.folder / f"{prefix}-label.nii.gz"
            color_path = self.folder / f"{prefix}_ColorTable.csv"
            nrrd_path  = self.folder / f"{prefix}.seg.nrrd"

            colors = _load_color_table(color_path) if color_path.exists() else {}

            self.segmentations[seg_name] = {
                "colors":  colors,
                "mode": settings["mode"],
                "label_path": label_path if label_path.exists() else None,
                "nrrd_path":  nrrd_path  if nrrd_path.exists()  else None,
            }

    def get_volume(self, seq_name: str) -> np.ndarray:
        """Return 3-D numpy array for a sequence. Raises KeyError if missing."""
        image_path = self.sequences.get(seq_name)
        if image_path is None:
            raise KeyError(f"Sequence '{seq_name}' not available for {self.id}")
        return nib.load(str(image_path)).get_fdata()

    def get_segmentation(self, seg_name: str) -> Tuple[np.ndarray, Dict]:
        """Return (label_array, color_dict) for a segmentation."""
        seg = self.segmentations.get(seg_name)
        if seg is None or seg["label_path"] is None:
            raise KeyError(f"Segmentation '{seg_name}' not available for {self.id}")
        return nib.load(str(seg["label_path"])).get_fdata(), seg["colors"]
    
    def info(self):
        """Load files on demand and print a structured data summary."""
        print(f"\n{'═' * 55}")
        print(f"  Patient : {self.id}")
        print(f"  Folder  : {self.folder}")
        print(f"{'═' * 55}")
        print("  GEOMETRY TOLERANCE")
        print(f"  Strict PyRadiomics default : {STRICT_GEOMETRY_TOLERANCE:.0e}")
        print(f"  Current practical check   : {GEOMETRY_TOLERANCE:.0e}; "
              "use the smallest value covering the observed drift.")

        print("\n  RAW SEQUENCES")
        print(f"  {'Name':<16} {'Shape (LR,AP,SI)':<20} {'Plane':<10} "
              f"{'Vox size LR/AP/SI (mm)':<27} {'dtype / bits / range'}")
        print(f"  {'─' * 115}")
        for name, image_path in self.sequences.items():
            if image_path is None:
                print(f"  {name:<16} — not found")
                continue
            image = nib.load(str(image_path))
            vox = _medical_voxel_sizes(image)
            medical_shape, axis_labels, plane = _medical_geometry(image)
            print(f"  {name:<16} {str(medical_shape):<20} {plane:<10} "
                f"{vox[0]:.2f} × {vox[1]:.2f} × {vox[2]:.2f}   "
                f"{_data_metadata(image)}")

        print("\n  SEGMENTATIONS")
        print(f"  {'Name':<12} {'Mask values':<12} {'Image shape LR/AP/SI':<21} "
            f"{'Mask shape LR/AP/SI':<21} {'Plane':<10} {'Shape':<7} {'Geometry':<8}")
        print(f"  {'─' * 110}")
        for name, seg in self.segmentations.items():
            label_path = seg["label_path"]
            image_path = self.sequences.get(name)

            if label_path is None and image_path is None:
                print(f"  {name:<12} {'—':<12} {'—':<15} {'—':<15} "
                      f"{'N/A':<7} {'N/A':<8}")
                continue

            if label_path is None:
                image = nib.load(str(image_path))
                medical_shape, axis_labels, plane = _medical_geometry(image)
                print(f"  {name:<12} {'—':<12} {str(medical_shape):<21} "
                    f"{'—':<21} {plane:<10} {'N/A':<7} {'N/A':<8}")
                continue

            mask = nib.load(str(label_path))
            data = mask.get_fdata()
            mask_values = np.unique(data[data > 0]).astype(int).tolist()

            if image_path is None:
                medical_shape, axis_labels, plane = _medical_geometry(mask)
                print(f"  {name:<12} {str(mask_values):<12} {'—':<21} "
                    f"{str(medical_shape):<21} {plane:<10} {'N/A':<7} {'N/A':<8}")
                continue

            image = nib.load(str(image_path))
            image_shape, image_axes, image_plane = _medical_geometry(image)
            mask_shape, mask_axes, mask_plane = _medical_geometry(mask)
            shape_match = image.shape == mask.shape
            spacing_match = np.allclose(
                image.header.get_zooms()[:3],
                mask.header.get_zooms()[:3],
                    atol=GEOMETRY_TOLERANCE,
            )
            affine_match = np.allclose(
                image.affine, mask.affine, atol=GEOMETRY_TOLERANCE
            )
            recommended_tolerance = _recommended_geometry_tolerance(image, mask)
            shape_status = "MATCH" if shape_match else "MISMATCH"
            geometry_status = "MATCH" if spacing_match and affine_match else "MISMATCH"
            mismatch_details = []
            if not shape_match:
                mismatch_details.append(
                    f"shape image={image.shape} mask={mask.shape}"
                )
            if not spacing_match:
                mismatch_details.append(
                    f"spacing image={tuple(round(v, 3) for v in image.header.get_zooms()[:3])} "
                    f"mask={tuple(round(v, 3) for v in mask.header.get_zooms()[:3])}"
                )
            if not affine_match:
                affine_difference = float(np.max(np.abs(image.affine - mask.affine)))
                mismatch_details.append(f"affine max_diff={affine_difference:.4f}")
            details = f" | {'; '.join(mismatch_details)}" if mismatch_details else ""
            details += f" | recommended geometryTolerance={recommended_tolerance}"
            plane = image_plane if image_plane == mask_plane else f"{image_plane}/{mask_plane}"
            print(f"  {name:<12} {str(mask_values):<12} "
                f"{str(image_shape):<21} {str(mask_shape):<21} "
                f"{plane:<10} {shape_status:<7} {geometry_status:<8}{details}")
        print()
    
    def correct(self) -> int:
        modified = 0

        for name, seg in self.segmentations.items():
            label_path = seg["label_path"]
            image_path = self.sequences.get(name)

            if label_path is None or image_path is None:
                continue

            image = nib.load(str(image_path))
            mask = nib.load(str(label_path))

            img_shape = image.shape[:3]
            mask_shape = mask.shape[:3]

            if (
                img_shape == mask_shape
                and np.allclose(
                    image.affine, mask.affine,
                    atol=GEOMETRY_TOLERANCE
                )
            ):
                continue

            data = np.asarray(mask.dataobj)
            assert data.min() >= 0 and data.max() <= 255, (
                f"{label_path}: labels out of uint8 range"
            )
            data = np.rint(data).astype(np.uint8)

            source = mask

            if img_shape == mask_shape:
                # voxel array is reversed on axes 0 and 1.
                flipped_data = data[::-1, ::-1, :].copy()

                flip = np.eye(4)
                flip[0, 0] = -1
                flip[1, 1] = -1
                flip[0, 3] = mask_shape[0] - 1
                flip[1, 3] = mask_shape[1] - 1

                flipped_affine = mask.affine @ flip

                if np.allclose(
                    flipped_affine, image.affine,
                    atol=GEOMETRY_TOLERANCE
                ):
                    source = nib.Nifti1Image(
                        flipped_data,
                        flipped_affine
                    )
                else:
                    source = mask

            res = nimg.resample_img(
                source,
                target_affine=image.affine,
                target_shape=img_shape,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )

            new_data = np.rint(
                np.asarray(res.dataobj)
            ).astype(np.uint8)

            bak = Path(f"{label_path}.bak")
            if not bak.exists():
                shutil.copy2(label_path, bak)

            out = nib.Nifti1Image(new_data, image.affine)
            out.set_qform(image.affine, code=1)
            out.set_sform(image.affine, code=1)
            out.set_data_dtype(np.uint8)

            nib.save(out, str(label_path))
            modified += 1

        return modified

    def available_sequences(self) -> List[str]:
        """Return sequences for which an image was found."""
        return [
            name for name, path in self.sequences.items()
            if path is not None
        ]

    def available_segmentations(self) -> List[str]:
        """Return segmentations for which a label image was found."""
        return [
            name for name, seg in self.segmentations.items()
            if seg["label_path"] is not None
        ]

    def get_image_path(self, sequence: str) -> Path:
        data = self.sequences.get(sequence)
        return data

    def get_segmentation_path(self, region: str) -> Path:
        data = self.segmentations.get(region)
        return data["label_path"] if data else None

    def to_radiomics_jobs(
        self,
        regions: Optional[List[str]] = None,
        mode: Optional[str] = None,
        labels: Optional[Dict[str, int]] = None,
        slice_selection=None,
    ) -> List[dict]:
        """
        Convert this Patient into PyRadiomics extraction jobs.

        Each job has the structure expected by RadiomicsExtractor:
            img_path
            seg_path
            label
            mode

        Plus metadata:
            patient_id
            sequence
            region
        """

        if regions is None:
            regions = list(SEG_PREFIXES.keys())
        labels = labels or {}
        jobs = []
        for region in regions:
            seg = self.segmentations.get(region)
            if seg is None or seg["label_path"] is None:
                continue
            sequence = region
            image = self.sequences.get(sequence)
            if image is None:
                continue
            image_path = self.get_image_path(sequence)
            seg_path = self.get_segmentation_path(region)
            job = {
                "img_path": image_path,
                "seg_path": seg_path,
                "mode": mode or seg["mode"],

                # metadata passed through unchanged by RadiomicsExtractor
                "patient_id": self.id,
                "sequence": sequence,
                "region": region,
            }

            if region in labels:
                job["label"] = labels[region]

            if slice_selection is not None:
                job["slice_selection"] = slice_selection

            jobs.append(job)

        return jobs


class PatientDataset:

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

        self.patients = [
            Patient(folder)
            for folder in sorted(self.root.iterdir())
            if folder.is_dir()
        ]

    def to_radiomics_jobs(
        self,
        regions: Optional[List[str]] = None,
        mode: Optional[str] = None,
        labels: Optional[Dict[str, int]] = None,
        slice_selection=None,
    ) -> List[dict]:

        jobs = []

        for patient in self.patients:

            patient_jobs = patient.to_radiomics_jobs(
                regions=regions,
                mode=mode,
                labels=labels,
                slice_selection=slice_selection,
            )

            jobs.extend(patient_jobs)

        return jobs

    def __getitem__(self, iterator: str) -> Patient:
        """Return the Patient object for a given patient ID."""
        return self.patients[iterator]
    
    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self):
        return iter(self.patients)

    def correct_geometry(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        regions: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> List[Dict]:
        """Correct all patient masks, optionally overwriting original files."""
        reports = []
        for patient in self:
            reports.extend(
                patient.correct_all_geometry(
                    output_dir=output_dir,
                    regions=regions,
                    overwrite=overwrite,
                )
            )
        return reports