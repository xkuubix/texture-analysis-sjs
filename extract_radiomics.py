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
    print(f"Using configuration file: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Regions: {args.regions}")
    print(f"Label: {args.label}")
    print(f"Slice selection: {args.slice_selection}")
    print(f"Parameter file 3D: {args.param_file_3d}")
    print(f"Parameter file 2D: {args.param_file_2d}")
    print(f"Parameter file ADC: {args.param_file_adc}")
    print(f"Use N4: {args.use_n4}\n")
    
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    dataset = PatientDataset(args.data_root, args.use_n4)

    if args.correct:
        # for patient in dataset.patients:
        #     patient.correct()
        for patient in dataset.patients:
            patient.info()

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
    param_file_adc = args.param_file_adc if args.param_file_adc.exists() else None
    extractor = RadiomicsExtractor(
        param_file_3d=str(param_file_3d) if param_file_3d else None,
        param_file_2d=str(param_file_2d) if param_file_2d else None,
        param_file_overrides={
            "ADC": str(param_file_adc) # for ADC sequences have prio over 2D and 3D param files
        } if param_file_adc else None,
        default_mode=args.mode or "3D",
    )
    results = extractor.extract(jobs)
    label_str = (
        "_".join(map(str, args.label))
        if isinstance(args.label, (list, tuple))
        else str(args.label)
    )

    _write_results(
        results,
        args.output_dir / f"v3_radiomics_results_label_{label_str}.csv"
    )
    print(f"Wrote {len(results)} results to {args.output_dir / f'v3_radiomics_results_label_{label_str}.csv'}")
    errors = [result for result in results if result.get("error")]
    print(f"Rows with errors: {len(errors)}")
    for result in errors:
        print(f"  {result.get('region')}: {result['error']}")


if __name__ == "__main__":
    main()
#
# %%