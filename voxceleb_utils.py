import os.path as osp
import glob

def read_Vox_lines(file):
    Tlines, Flines = [], []	
    with open(file) as f_in:
        while True:
            line = f_in.readline().strip()					
            if not line:
                break
            if int(line[0]):
                Tlines.append(line)
            else:
                Flines.append(line)
    return Tlines, Flines


def get_all_videos_path(root_dir, voxceleb_list_file):
    Tlines, _ = read_Vox_lines(voxceleb_list_file)
    all_videos_path = [osp.join(root_dir, f"{line.split()[2][:-4]}.mp4") for line in Tlines]
    return all_videos_path


def get_all_videos_path_v2(root_dir, dev_file_path):
    lines = open(dev_file_path).read().splitlines()
    all_videos_path = [osp.join(root_dir, f"{line}.mp4") for line in lines]
    return all_videos_path


def save_voxceleb_list(voxceleb_list_file, train_lines=26000):
    Tlines, _ = read_Vox_lines(voxceleb_list_file)
    Tlines = [line.split()[2][:-4] for line in Tlines]

    print(len(Tlines), Tlines[0])
    train_split = '\n'.join(Tlines[:train_lines])
    val_split = '\n'.join(Tlines[train_lines:])
    
    train_file = open("train.txt", "w")
    train_file.write(train_split)

    val_file = open("val.txt", "w")
    val_file.write(val_split)


def save_voxceleb_list_from_folders(root_dir):
    files = glob.glob(f'{root_dir}/*/*/*/000000.jpg')
    lines_list = ['/'.join(line.split('/')[-4:-1]) for line in files]

    print(len(lines_list), lines_list[:5])

    train_split = '\n'.join(lines_list)
    train_file = open("dev.txt", "w")
    train_file.write(train_split)

    
if __name__ == "__main__":
    # get_all_videos_path("/home/zhanghm/Datasets/VoxCeleb2/dev/mp4", 
    #                     "/home/zhanghm/Research/Face/TalkNet_ASD/TalkSet/lists/lists_in/Vox_list.txt")

    # save_voxceleb_list("/home/zhanghm/Research/Face/TalkNet_ASD/TalkSet/lists/lists_in/Vox_list.txt")

    # save_voxceleb_list_from_folders("/data/data0/zhanghm/VoxCeleb2/Wav2LipSet")

    get_all_videos_path_v2("/home/zhanghm/Datasets/VoxCeleb2/dev/mp4", "dev.txt")

