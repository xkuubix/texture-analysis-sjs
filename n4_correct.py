# %%
import logging
from data_utils import PatientDataset
from cli_args import parse_args


def main():
    args = parse_args()
    print(f"Using configuration file: {args.config}")
    print(f"Data root: {args.data_root}")

    assert args.data_root.exists(), f"Data root does not exist: {args.data_root}"
    assert not args.use_n4, "use_n4 argument must be False, as this script is intended to apply N4 correction to the original images."
    

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    dataset = PatientDataset(args.data_root, args.use_n4)
    for patient in dataset:
        for k, v in patient.sequences.items():
            assert "N4" not in str(v), f"N4 have been already applied to {k}"
    for patient in range(len(dataset.patients)):
        dataset[patient].n4_correct(overwrite=False)

if __name__ == "__main__":
    main()
#
# %%