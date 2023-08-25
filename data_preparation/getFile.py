from google.colab import drive
import zipfile
import os

def extract_zip_to_directory(zip_filename, extract_dir):
    # Mount Google Drive to access files, comment out after the first execution
    if not os.path.exists('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive')

    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction complete.")