
import os
import shutil

"""Traverse nested folders to find *-t1c.nii.gz files and copy them to a mirrored folder structure."""

def extract_t1c_files(root_dir, output_dir, file_suffix='-seg.nii.gz'):
    """
    Traverse nested folders to find *-t1c.nii.gz files and copy them to a mirrored folder structure.

    Args:
        root_dir (str): Path to the top-level directory containing patient folders.
        output_dir (str): Destination path for extracted files.
        file_suffix (str): File suffix to look for (default: '-t1c.nii.gz').
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(file_suffix):
                # Full path to source file
                src_file_path = os.path.join(dirpath, filename)

                # Relative path from root
                rel_path = os.path.relpath(src_file_path, root_dir)

                # Construct destination path
                dest_file_path = os.path.join(output_dir, rel_path)

                # Create directory if not exist
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

                # Copy file
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {rel_path}")

    print(f"\n Done. Extracted files saved to: {output_dir}")


root_data_dir = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa"
output_data_dir = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-seg_only"

extract_t1c_files(root_data_dir, output_data_dir)

