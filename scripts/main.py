
import subprocess

def run_script(path):
    print(f"Running: {path}")
    result = subprocess.run(["python", path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(" Errors:")
        print(result.stderr)

if __name__ == "__main__":
    scripts = [
        "scripts/extract_t1c_files_1modality.py",
        "scripts/img_text_pair.py",
        "scripts/nifti3d_to_2dpng.py",
        "scripts/dataloader_&_finetuning_pipeline.py"
    ]

    for script in scripts:
        run_script(script)

    print("All stages completed successfully.")
