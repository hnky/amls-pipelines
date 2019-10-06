from azureml.core import Workspace
from azureml.core import Run
from azureml.core import Experiment
import argparse
import json
import os
import numpy as np
import cv2
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_path',
        type=str,
        default='',
        help='Variant name you want to give to the model.'
    )
    parser.add_argument(
        '--destination_path',
        type=str,
        default='',
        help='Location of trained model.'
    )
    parser.add_argument(
        '--pic_size',
        type=int,
        default='',
        help='Location of trained model.'
    )

    args,unparsed = parser.parse_known_args()
    print('Source Path '+args.source_path)
    print('Destination Path '+args.destination_path)

    map_characters = {0:"marge_simpson", 1: "homer_simpson" }
    source_path = args.source_path
    pic_size = args.pic_size

    run = Run.get_context()
    pipeline_run_Id = run._root_run_id
   
    print("Pipeline Run Id: ",pipeline_run_Id)

    # Create the output folder
    destination_path = args.destination_path
    os.makedirs(destination_path, exist_ok=True)
    
    for k, char in map_characters.items():
        # Create the output directory on the destination storage
        output_char_path = os.path.join(destination_path,char)
        os.makedirs(output_char_path, exist_ok=True)
        print('Output Char Path ',output_char_path)

        # Get all the pictures
        pictures = [k for k in glob.glob(os.path.join(source_path,"characters/%s/*" % char))]

        # Take random 1000 pictures
        # Resize the pictures
        # Save the resized pictures to the output folder
        for pic in np.random.choice(pictures, 1000):
            a = cv2.imread(pic)
            a = cv2.resize(a, (pic_size,pic_size))
            cv2.imwrite(os.path.join(output_char_path,os.path.basename(pic)), a)