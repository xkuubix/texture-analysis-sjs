import os
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union

import numpy as np
import SimpleITK as sitk
from tqdm import trange
from radiomics import featureextractor
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

logger = logging.getLogger(__name__)

class RadiomicsExtractor:
    """
    Extracts radiomics features from grayscale medical images (MRI, CT, ...).

    Supports two extraction modes:
      - "3D": whole-volume extraction, texture matrices computed in 3D.
        Use for volumetric organ/lesion segmentations.
      - "2D": either (a) PyRadiomics' native per-slice mode (`force2D=True`
        in the parameter file, features averaged across in-plane slices of
        a 3D volume), or (b) a single representative 2D slice extracted
        from the volume (e.g. the slice with the largest ROI area) and
        treated as a plain 2D image. Both are driven from the SAME method
        here - which one you get depends on your 2D parameter file and,
        optionally, the `slice_selection` you pass per case.

    A separate PyRadiomics parameter file is expected for each mode, since
    2D and 3D settings (force2D, kernel radius, resampling, etc.) differ.

    Args:
        param_file_3d: path to PyRadiomics param YAML/JSON for 3D extraction.
        param_file_2d: path to PyRadiomics param YAML/JSON for 2D extraction
            (typically has `force2D: true`, `force2Ddimension: 0/1/2`).
        transforms: optional callable(image: np.ndarray, mask: np.ndarray)
            -> (image, mask), applied before feature extraction. Use for
            resampling/normalization/cropping - no hair removal or other
            skin-image-specific preprocessing here.

    Input dicts passed to extract_radiomics / *_extraction expect:
        {
            "img_path": str | Path | sitk.Image,
            "seg_path": str | Path | sitk.Image,
            "label": int (optional - overrides the extractor's configured
                          label, e.g. to select "right parotid" vs "left"),
            "mode": "2D" | "3D" (optional - defaults to self.default_mode),
            "slice_selection": "largest" | int (optional, 2D mode only) -
                          "largest" picks the axial slice with the most
                          labeled voxels; an int picks that slice index
                          directly. If omitted, the whole volume is passed
                          through to the extractor as-is (native force2D
                          behaviour from the param file).
            ... any extra keys (e.g. "patient_id", "sequence", "region")
            are passed through untouched into the returned dict, so you
            can trace every row of results back to its source.
        }
    """

    def __init__(
        self,
        param_file_3d: Optional[str] = None,
        param_file_2d: Optional[str] = None,
        transforms: Optional[Callable] = None,
        default_mode: str = "3D",
    ):
        if param_file_3d is None and param_file_2d is None:
            raise ValueError("Provide at least one of param_file_3d / param_file_2d")

        self.extractors: Dict[str, featureextractor.RadiomicsFeatureExtractor] = {}
        if param_file_3d is not None:
            self.extractors["3D"] = featureextractor.RadiomicsFeatureExtractor(param_file_3d)
        if param_file_2d is not None:
            self.extractors["2D"] = featureextractor.RadiomicsFeatureExtractor(param_file_2d)

        self.default_mode = default_mode
        self.transforms = transforms

        for mode, ex in self.extractors.items():
            msg = f"\n\n[{mode}] Enabled Image Types: {list(ex.enabledImagetypes.keys())}"
            msg += f"\n[{mode}] Enabled Features: {list(ex.enabledFeatures.keys())}"
            logger.info(msg)
        if self.transforms:
            logger.info(f"Transforms: {self.transforms}")

    # ── public API ──────────────────────────────────────────────────────

    def get_enabled_image_types(self, mode: Optional[str] = None) -> List[str]:
        return list(self._extractor_for(mode).enabledImagetypes.keys())

    def get_enabled_features(self, mode: Optional[str] = None) -> List[str]:
        return list(self._extractor_for(mode).enabledFeatures.keys())

    def extract_radiomics(self, d: dict) -> dict:
        """Extract features for one (image, segmentation) pair. Never raises -
        on failure, returns the passthrough metadata plus an 'error' key, so
        one bad case doesn't kill a parallel batch."""
        meta = {k: v for k, v in d.items()
                if k not in ("img_path", "seg_path", "label", "mode", "slice_selection")}
        mode = d.get("mode", self.default_mode)
        slice_index = None
        slice_axis = None
        original_force2d_dimension = None
        force2d_dimension_changed = False
        try:
            extractor = self._extractor_for(mode)

            image = self._as_sitk_image(d["img_path"])
            mask = self._as_sitk_image(d["seg_path"])
            self._validate_geometry(image, mask, meta)

            label = d.get("label", extractor.settings.get("label", 1))

            if mode == "2D" and d.get("slice_selection") is not None:
                image, mask, np_axis, slice_index = self._select_slice(
                    image, mask, label, d["slice_selection"]
                )
                slice_axis = 2 - np_axis
                original_force2d_dimension = extractor.settings.get("force2Ddimension")
                extractor.settings["force2Ddimension"] = np_axis
                force2d_dimension_changed = True

            if self.transforms:
                image_geometry = image
                mask_geometry = mask
                im_arr = sitk.GetArrayFromImage(image)
                mk_arr = sitk.GetArrayFromImage(mask)
                im_arr, mk_arr = self.transforms(im_arr, mk_arr)
                image = sitk.GetImageFromArray(im_arr)
                mask = sitk.GetImageFromArray(mk_arr)
                image.CopyInformation(image_geometry)
                mask.CopyInformation(mask_geometry)
                image.CopyInformation(self._as_sitk_image(d["img_path"]))
                mask.CopyInformation(self._as_sitk_image(d["seg_path"]))

            features = extractor.execute(image, mask, label=label)
            return {
                **meta,
                "mode": mode,
                "label": label,
                "slice_index": slice_index,
                "slice_axis": slice_axis,
                **features,
            }

        except Exception as e:
            logger.error(f"Extraction failed for {meta}: {e}")
            return {
                **meta,
                "mode": mode,
                "label": d.get("label"),
                "slice_index": slice_index,
                "slice_axis": slice_axis,
                "error": str(e),
            }
        finally:
            if force2d_dimension_changed:
                extractor.settings["force2Ddimension"] = original_force2d_dimension

    def extract(self, list_of_dicts: List[dict]) -> List[dict]:
        logger.info("Extraction mode: serial")
        start_time = time.time()
        results = [self.extract_radiomics(list_of_dicts[i]) for i in trange(len(list_of_dicts))]
        self._log_elapsed(start_time)
        return results

    # ── internals ───────────────────────────────────────────────────────

    def _extractor_for(self, mode: Optional[str]) -> featureextractor.RadiomicsFeatureExtractor:
        mode = mode or self.default_mode
        if mode not in self.extractors:
            raise ValueError(f"No parameter file configured for mode={mode!r}. "
                              f"Available: {list(self.extractors.keys())}")
        return self.extractors[mode]

    @staticmethod
    def _as_sitk_image(img: Union[str, Path, sitk.Image]) -> sitk.Image:
        if isinstance(img, sitk.Image):
            return img
        path = str(img)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return sitk.ReadImage(path)

    @staticmethod
    def _validate_geometry(image: sitk.Image, mask: sitk.Image, meta: dict):
        """PyRadiomics requires image/mask to share size, spacing and
        direction - fail loudly and early with useful context rather than
        letting PyRadiomics raise an opaque error deep inside C++."""
        if image.GetSize() != mask.GetSize():
            raise ValueError(
                f"Image/mask size mismatch for {meta}: "
                f"image={image.GetSize()} mask={mask.GetSize()}"
            )
        if not np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-3):
            raise ValueError(
                f"Image/mask spacing mismatch for {meta}: "
                f"image={image.GetSpacing()} mask={mask.GetSpacing()}"
            )
        if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-3):
            raise ValueError(
                f"Image/mask origin mismatch for {meta}: "
                f"image={image.GetOrigin()} mask={mask.GetOrigin()}"
            )
        if not np.allclose(image.GetDirection(), mask.GetDirection(), atol=1e-3):
            raise ValueError(
                f"Image/mask direction mismatch for {meta}: "
                f"image={image.GetDirection()} mask={mask.GetDirection()}"
            )

    @staticmethod
    def _select_slice(image: sitk.Image, mask: sitk.Image, label: int,
                       slice_selection: Union[str, int]):
        """Return a 3D image/mask pair cropped to one representative slice."""
        spacing = np.asarray(mask.GetSpacing())
        if np.ptp(spacing) < 0.1 * spacing.max():
            sitk_axis = 2
        else:
            sitk_axis = int(np.argmax(spacing))
        np_axis = 2 - sitk_axis
        mask_arr = sitk.GetArrayFromImage(mask)
        counts = np.moveaxis(mask_arr == label, np_axis, 0).reshape(
            mask_arr.shape[np_axis], -1
        ).sum(axis=1)

        if slice_selection == "largest":
            if counts.max() == 0:
                raise ValueError(f"Label {label} has zero voxels in this mask; "
                                  f"cannot select a representative slice.")
            idx = int(np.argmax(counts))
        elif isinstance(slice_selection, int):
            idx = slice_selection
            if idx < 0 or idx >= mask.GetSize()[sitk_axis]:
                raise IndexError(
                    f"Slice index {idx} is out of bounds for axis {sitk_axis} "
                    f"with size {mask.GetSize()[sitk_axis]}"
                )
        else:
            raise ValueError(f"Unrecognised slice_selection: {slice_selection!r}")

        size = list(image.GetSize())
        index = [0, 0, 0]
        size[sitk_axis] = 1
        index[sitk_axis] = idx
        return (
            sitk.RegionOfInterest(image, size, index),
            sitk.RegionOfInterest(mask, size, index),
            np_axis,
            idx,
        )

    @staticmethod
    def _log_elapsed(start_time: float):
        dt = time.time() - start_time
        h, m, s = int(dt // 3600), int((dt % 3600) // 60), int(dt % 60)
        logger.info(f"Time taken: {h}h:{m}m:{s}s")