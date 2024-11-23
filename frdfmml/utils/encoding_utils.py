import chardet


def detect_encoding(file_path, sample_size=10 * 1024 * 1024):
    with open(file_path, 'rb') as file:
        # Read the first 10MB of the file
        sample = file.read(sample_size)

    # Detect the encoding using chardet
    result = chardet.detect(sample)

    return result['encoding']
