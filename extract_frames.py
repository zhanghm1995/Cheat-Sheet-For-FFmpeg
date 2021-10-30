import subprocess
import glob
from tqdm import tqdm
import os
import os.path as osp
import multiprocessing
import itertools

g_save_dir = ""


def extract_frames_from_single_video(fname, confidence):
    dirname = osp.dirname(fname)
    os.makedirs(dirname, exist_ok=True)

    nDataLoaderThread = 8
    output_dir = osp.join(g_save_dir, '%06d.jpg')
    command = f"ffmpeg -y -i {fname} -qscale:v 2 -threads {nDataLoaderThread} -f image2 {output_dir} -loglevel panic"
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
    input_dir = "/data/data0/zhanghm/LRS2"

    files = glob.glob(f'{input_dir}/pretrain/*/*.mp4')
    print(len(files))
    print(files[:10])

    # extract_frames_multiple_threads(files, 64)