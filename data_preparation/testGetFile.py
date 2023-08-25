import os
import tempfile
import unittest
import shutil 
import zipfile
import getFile

# Function to check if Google Drive is mounted
def is_google_drive_mounted():
    return os.path.exists('/content/drive/MyDrive')
    
def create_extraction_directory(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)

class TestGetFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create test_function_data directory
        self.test_function_data_dir = os.path.join(self.temp_dir, 'test_function_data')
        os.makedirs(self.test_function_data_dir)

        # Create test.zip with data.csv inside
        zip_file_path = os.path.join(self.test_function_data_dir, 'test.zip')
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            zipf.writestr('data.csv', '')  # Creating an empty data.csv file

    def tearDown(self):
        # Clean up the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
        
    def testExtractionExists(self):
        self.assertIsNotNone(getFile.extract_zip_to_directory)
        
    def test_google_drive_mounting(self):
        self.assertTrue(is_google_drive_mounted(), "Google Drive is not mounted!")
    
    def test_extraction_directory_creation(self):
        extract_dir = 'test_extraction_dir'
        create_extraction_directory(extract_dir)
        
        self.assertTrue(os.path.exists(extract_dir))
        
        # Clean up: Remove the created directory after the test
        os.rmdir(extract_dir)
        
    def test_basic_extraction(self):
        root_path = '/content/drive/MyDrive/final/data_preparation/'
        zip_filename = os.path.join(self.test_function_data_dir, 'test.zip') 
        extract_dir = tempfile.mkdtemp()
    
        try:
            getFile.extract_zip_to_directory(zip_filename, extract_dir)
            self.assertTrue(os.path.exists(extract_dir))
        finally:
            shutil.rmtree(extract_dir)

    def test_nonexistent_zip(self):
        root_path = '/content/drive/Mydrive/final/data_preparation/'
        zip_filename = os.path.join(self.test_function_data_dir, 'non_existent.zip')
        extract_dir = tempfile.mkdtemp()

        with self.assertRaises(FileNotFoundError):
            getFile.extract_zip_to_directory(zip_filename, extract_dir)

            # Clean up the temporary directory
            shutil.rmtree(extract_dir)

if __name__ == '__main__':
    unittest.main()
