import unittest
import os
from tempfile import NamedTemporaryFile

# Assume detect_encoding is defined in encoding_utils.py
from .encoding_utils import detect_encoding

class TestEncodingUtils(unittest.TestCase):

    def test_detect_encoding_utf8(self):
        # Create a temporary file with UTF-8 content
        with NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write('This is a test file with UTF-8 encoding.  包含中文字符。')
            temp_file_path = temp_file.name

        # Test if detect_encoding correctly detects UTF-8 encoding
        detected_encoding = detect_encoding(temp_file_path)
        self.assertEqual('utf-8', detected_encoding)

        # Cleanup the temporary file
        os.remove(temp_file_path)

    def test_detect_encoding_latin1(self):
        # Create a temporary file with Latin-1 (ISO-8859-1) content
        with NamedTemporaryFile(delete=False, mode='w', encoding='latin-1') as temp_file:
            temp_file.write('This is a test file with Latin-1 encoding. C\'est un test avec des caractères accentués, como ñ, ç, ü, and é.')
            temp_file_path = temp_file.name

        # Test if detect_encoding correctly detects Latin-1 encoding
        detected_encoding = detect_encoding(temp_file_path)
        self.assertEqual('ISO-8859-1', detected_encoding)

        # Cleanup the temporary file
        os.remove(temp_file_path)

if __name__ == '__main__':
    unittest.main()
