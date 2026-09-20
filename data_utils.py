
import re
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import csv
import nibabel as nib
import nilearn.image as nimg


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
    "T2_SPACE":      r"t[12]_space_tra.*iso_1\.0mm",
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
    for f in folder.iterdir():
        if f.is_file() and "-label.nii.gz" not in f.name and re.search(pattern, f.name, re.IGNORECASE):
            return f
    return None


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

        print("\n  RAW SEQUENCES")
        print(f"  {'Name':<16} {'Shape':<20} {'Vox size (mm)'}")
        print(f"  {'─' * 52}")
        for name, image_path in self.sequences.items():
            if image_path is None:
                print(f"  {name:<16} — not found")
                continue
            image = nib.load(str(image_path))
            vox = image.header.get_zooms()[:3]
            print(f"  {name:<16} {str(image.shape):<20} "
                  f"{vox[0]:.2f} × {vox[1]:.2f} × {vox[2]:.2f}")

        print("\n  SEGMENTATIONS")
        print(f"  {'Name':<12} {'Mask values':<12} {'Image shape':<15} "
              f"{'Mask shape':<15} {'Shape':<7} {'Geometry':<8}")
        print(f"  {'─' * 75}")
        for name, seg in self.segmentations.items():
            label_path = seg["label_path"]
            image_path = self.sequences.get(name)

            if label_path is None and image_path is None:
                print(f"  {name:<12} {'—':<12} {'—':<15} {'—':<15} "
                      f"{'N/A':<7} {'N/A':<8}")
                continue

            if label_path is None:
                image = nib.load(str(image_path))
                print(f"  {name:<12} {'—':<12} {str(image.shape):<15} "
                      f"{'—':<15} {'N/A':<7} {'N/A':<8}")
                continue

            mask = nib.load(str(label_path))
            data = mask.get_fdata()
            mask_values = np.unique(data[data > 0]).astype(int).tolist()

            if image_path is None:
                print(f"  {name:<12} {str(mask_values):<12} {'—':<15} "
                      f"{str(mask.shape):<15} {'N/A':<7} {'N/A':<8}")
                continue

            image = nib.load(str(image_path))
            shape_match = image.shape == mask.shape
            spacing_match = np.allclose(
                image.header.get_zooms()[:3],
                mask.header.get_zooms()[:3],
                atol=1e-3,
            )
            affine_match = np.allclose(image.affine, mask.affine, atol=1e-3)
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
            print(f"  {name:<12} {str(mask_values):<12} "
                  f"{str(image.shape):<15} {str(mask.shape):<15} "
                  f"{shape_status:<7} {geometry_status:<8}{details}")
        print()

    def correct(self) -> None:
            """
            Check image/segmentation geometry alignment for each segmentation and 
            overwrite the segmentation NIfTI file on disk if a mismatch is detected.
            """

            for name, seg in self.segmentations.items():
                label_path = seg["label_path"]
                image_path = self.sequences.get(name)

                if label_path is None or image_path is None:
                    continue

                image = nib.load(str(image_path))
                mask = nib.load(str(label_path))

                shape_match = image.shape == mask.shape
                spacing_match = np.allclose(
                    image.header.get_zooms()[:3],
                    mask.header.get_zooms()[:3],
                    atol=1e-3,
                )
                affine_match = np.allclose(image.affine, mask.affine, atol=1e-3)

                if shape_match and spacing_match and affine_match:
                    continue
                print(f"\n{'═' * 55}")
                print(f"  CORRECTING GEOMETRY: {self.id}")
                print(f"  Folder : {self.folder}")
                print(f"{'═' * 55}")
                print(f"  {name:<12} MISMATCH detected -> Overwriting...")

                # Scenario A: Same dimensions, update header/affine metadata
                if shape_match:
                    print("    └─ Copying affine and header from original image...")
                    mask_data = mask.get_fdata().astype(np.uint8)
                    corrected_mask = nib.Nifti1Image(mask_data, affine=image.affine, header=image.header)
                    corrected_mask.set_data_dtype(np.uint8)
                    nib.save(corrected_mask, str(label_path))
                    print(f"    └─ Overwritten: {label_path.name}")

                # Scenario B: Shape mismatch, warn user
                else:
                    print("    └─ Shape mismatch, cannot align...")
            print()

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