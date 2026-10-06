# %%
import csv
import logging
from pathlib import Path

from RadiomicsExtractor import RadiomicsExtractor
from data_utils import SEG_PREFIXES, PatientDataset
from cli_args import parse_args
logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

def _write_results(results, output_path: Path):
    """Write heterogeneous PyRadiomics result dictionaries to one CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_columns = [
        "patient_id", "region", "sequence", "mode", "label",
        "slice_index", "slice_axis", "error",
    ]
    remaining = sorted({key for result in results for key in result} - set(first_columns))
    fieldnames = first_columns + remaining
    if not fieldnames:
        raise ValueError("No extraction results were produced")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    dataset = PatientDataset(args.data_root)

    if args.correct:
        for patient in dataset.patients:
            patient.correct()

    regions = args.regions
    label_regions = regions or list(SEG_PREFIXES)
    jobs = dataset.to_radiomics_jobs(
        regions=regions,
        mode=args.mode,
        labels={region: args.label for region in label_regions},
        slice_selection=args.slice_selection,
    )
    if not jobs:
        raise RuntimeError(f"No usable image/segmentation pairs found under {args.data_root}")

    param_file_3d = args.param_file_3d if args.param_file_3d.exists() else None
    param_file_2d = args.param_file_2d if args.param_file_2d.exists() else None
    extractor = RadiomicsExtractor(
        param_file_3d=str(param_file_3d) if param_file_3d else None,
        param_file_2d=str(param_file_2d) if param_file_2d else None,
        default_mode=args.mode or "3D",
    )
    results = extractor.extract(jobs)

    _write_results(results, args.output)
    print(f"Wrote {len(results)} results to {args.output}")
    errors = [result for result in results if result.get("error")]
    print(f"Rows with errors: {len(errors)}")
    for result in errors:
        print(f"  {result.get('region')}: {result['error']}")


if __name__ == "__main__":
    main()

# %%
