import tempfile


if __name__ == "__main__":
    folder = tempfile.mkdtemp(dir="./")
    print(folder)
