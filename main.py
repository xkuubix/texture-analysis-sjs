import radiomics
import nibabel as nib
import pydicom


def main():
    print("Radiomics version:", radiomics.__version__)
    print("Nibabel version:", nib.__version__)
    print("Pydicom version:", pydicom.__version__)
    

if __name__ == "__main__":
    main()
