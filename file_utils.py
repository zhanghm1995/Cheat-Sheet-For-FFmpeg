import glob
import os


def get_directories_num(root_dir):
    folder_num = 0
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            for sub_dir in os.scandir(entry):
                for sub_dir2 in os.scandir(sub_dir):
                    if sub_dir2.is_dir():
                        folder_num += 1
    
    print(folder_num)

def get_num(root_dir):
    print(len(glob.glob(os.path.join(root_dir, "**"), recursive=True)) - 1)


def get_num2(root_dir):
    for base, dirs, files in os.walk(root_dir):
        print(len(dirs), dirs)


def get_directories_num2(root_dir):
    files = glob.glob(f'{root_dir}/*/*/000000.jpg')
    print(len(files), files[:5])

if __name__ == "__main__":
    input_dir = "/data/data0/zhanghm/VoxCeleb2/Wav2LipSet"
    input_dir = "/data/data0/zhanghm/AudioVision/Wav2LipTEDHQ/wav2lipTedHQ_preprocessed/val"
    get_directories_num2(input_dir)

    # get_num2("/data/data0/zhanghm/VoxCeleb2/Wav2LipSet/id02706")