import subprocess
import glob
from tqdm import tqdm
import os
import os.path as osp
import multiprocessing
import itertools
import cv2

g_save_root = "/data/zhanghm/LRS2"


def extract_frames_from_single_video(fname, confidence=None):
    dirname = '/'.join(fname.split('/')[-2:])
    dirname = osp.join(g_save_root, dirname[:-4]) ## remove the .mp4
    
    os.makedirs(dirname, exist_ok=True)
    video = cv2.VideoCapture(fname)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    if fps != 25: ## If FPS is not 25, we should return this file
        print(f"{fname} FPS is not 25 it's {fps}")

    nDataLoaderThread = 16
    output_dir = osp.join(dirname, '%06d.jpg')
    command = f"ffmpeg -y -i {fname} -start_number 000000 -qscale:v 2 -threads {nDataLoaderThread} -f image2 {output_dir} -loglevel panic"
    out = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)

    if out != 0:
        raise ValueError(f'Conversion failed {fname}.')


def extract_frames_multiple_threads(files_list, number_of_cpus):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    confidence = 0.7

    function_parameters = zip(
        files_list,
        itertools.repeat(confidence)
    )

    pool.starmap(extract_frames_from_single_video, function_parameters)
    print("Done")


if __name__ == "__main__":
    input_dir = "/data/zhanghm/LRS2_dataset/mvlrs_v1/main"

    files = glob.glob(f'{input_dir}/*/*.mp4')
    print(len(files))
    print(files[:3])

    extract_frames_multiple_threads(files, 64)
    # extract_frames_from_single_video(files[0])