import subprocess
import glob
from tqdm import tqdm
import os
import os.path as osp
import multiprocessing
import itertools


def extract_audios_from_videos(files_list, confidence):
    for fname in tqdm(files_list):
        outfile = fname.replace('.mp4', '.wav')
        outfile = outfile.replace('pretrain', 'audio/pretrain')
        dirname = osp.dirname(outfile)
        os.makedirs(dirname, exist_ok=True)

        out = subprocess.call('ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic >/dev/null 2>/dev/null' %(fname, 8, outfile), shell=True)
        if out != 0:
            raise ValueError(f'Conversion failed {fname}.')
    print("Done.")


def extract_audio_from_single_video(fname, confidence=None):
    outfile = fname.replace('.mp4', '.wav')
    outfile = outfile.replace('pretrain', 'audio/pretrain')
    dirname = osp.dirname(outfile)
    os.makedirs(dirname, exist_ok=True)

    out = subprocess.call('ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic >/dev/null 2>/dev/null' %(fname, 8, outfile), shell=True)
    if out != 0:
        raise ValueError(f'Conversion failed {fname}.')


g_save_root = "/data/zhanghm/LRS2"
def extract_audio_from_single_video2(fname, confidence=None):
    """Save the audio using the same 'audio.wav' file name
    """
    dirname = '/'.join(fname.split('/')[-2:])
    dirname = osp.join(g_save_root, dirname[:-4]) ## remove the .mp4
    output_dir = osp.join(dirname, 'audio.wav')

    out = subprocess.call('ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic >/dev/null 2>/dev/null' %(fname, 16, output_dir), shell=True)
    if out != 0:
        raise ValueError(f'Conversion failed {fname}.')


def extract_audios_multiple_threads(files_list, number_of_cpus):
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

    pool.starmap(extract_audio_from_single_video2, function_parameters)
    print("Done")


if __name__ == "__main__":
    ## ------For LRS3 dataset
    input_dir = "/data/data0/zhanghm/LRS3"
    files = glob.glob(f'{input_dir}/pretrain/*/*.mp4')
    ## ------For LRS2 dataset
    input_dir = "/data/zhanghm/LRS2_dataset/mvlrs_v1/main"
    files = glob.glob(f'{input_dir}/*/*.mp4')

    print(len(files))
    print(files[:10])

    # extract_audios_from_videos(files, input_dir)

    extract_audios_multiple_threads(files, 64)

    # extract_audio_from_single_video2(files[0]) ## For single file testing