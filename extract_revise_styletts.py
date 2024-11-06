import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tqdm import tqdm
import argparse
import glob
import re


def __cmd():
    description = "extract w2v"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--input_dir",
        type=str,
        default='',
        required=False,
        help="the audio corpus dir.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        raise ValueError(f"input_dir not exists: {input_dir}")
    
    txt_files = glob.glob(f'{input_dir}/**/transcription.txt.styletts.*', recursive=True)

    for item in tqdm(txt_files):
        # _meta_file = os.path.basename(meta_file).split(".")[0]
        # styletts_file = os.path.join(os.path.dirname(meta_file), "transcription.txt.styletts.train")
        save_path = re.sub(r'^/data',r'/data2', item).strip()

        with open(item, "r", encoding="utf-8") as ttf, open(save_path, "w", encoding="utf-8") as wf:
            for line in ttf:
                new_line = re.sub(r'^/data',r'/data2', line).strip()
                wf.write(new_line+"\n")

if __name__ == '__main__':
    __cmd()

'''
python extract_revise_styletts.py --input_dir /data/liuhuang/dataset/acoustic_train_dataset/collect_set/Xiaoyou
'''