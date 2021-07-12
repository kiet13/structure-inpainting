from imageio import imread, imwrite
from glob import glob
from PIL import Image
import numpy as np
import os
import time
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='input path')
parser.add_argument('--output_path', type=str, help='output path')
parser.add_argument('--Lambda', type=float, help='lambda argument for image smoothing via l0 gradient minimization')
args = parser.parse_args()

def run_octave(config, code):
    
    arg = ['octave']
    for k, v in config.items():
        arg.extend(['--eval', '%s=%s;' % (k, v)])

    arg.extend(['--eval', code])

    try:
        
        subprocess.check_output(arg, stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('octave failed')
        print('returncode:', e.returncode)
        print('output:', e.output)
        raise

def gen_L0_octave_code(Lambda):

    p1 = r"""
    Im = imread(input_path);
    [dir, name, ext] = fileparts(input_path);
    """ 

    p2 = "S = L0Smoothing(Im," + str(Lambda) + ");"
   
    p3 = r"""
    write_name = strrep(input_path, dir, output_path);
    [filepath,~,~] = fileparts(write_name);
    if ~exist(filepath, 'dir')
        mkdir(filepath);
    end
    imwrite(S, write_name);    
    """

    octave_code = p1 + p2 + p3
    return octave_code

def generate_structure_images(octave_code, input_path, output_path):
    input_list = glob(input_path+'/*')
    if os.path.isdir(output_path) == False:
        os.mkdir(output_path)

    for i, filepath in enumerate(input_list):
        config = dict(
            input_path  = "'%s'" % filepath,
            output_path = "'%s'" % output_path,
        )
        run_octave(config, octave_code)
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(input_list)} images")

def main():
    octave_code = gen_L0_octave_code(args.Lambda)
    generate_structure_images(octave_code, args.input_path, args.output_path)

if __name__ == '__main__':
    main()
    

