import os

base_path = "../../../data/upCam/raw/"


def rename_files_in_directory(directory: str) -> None:
    """Rename all 24 files in a day-directory to follow the same name scheme ("%02d-%02d.mp4" / DAY-HOUR").

    :param directory: Directory that will be processed.
    :return: None
    """
    file_count = 0
    for file_name in sorted(os.listdir(base_path + directory + "/")):
        old = base_path + directory + "/" + file_name
        new = base_path + directory + "/" + directory + "-%02d.mp4" % file_count
        os.rename(old, new)
        file_count += 1


if __name__ == '__main__':
    for i in range(10):
        rename_files_in_directory("%02d" % i)
