import subprocess

def count_lines_with_wc(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        line_count = int(result.stdout.split()[0])
        return line_count
    else:
        raise Exception(f"Error occurred: {result.stderr}")