# improved_DDNet

## installation

'git clone {this repo}'

'pip -r install requirements.txt'

## download train/test.pickle to folder "data", and the folder "pose_new_v2" which contains the skeleton file of all videos to root folder
./data/COBOT/train.pickle     
.data/COBOT/test.
./pose_new_v2

## Training
'python train.py  --run_name {run name} #  find the list of "run_name" in "models/DDNet_Original.py" '

## Online test
Visit 'online.py' and then set the value of 'exp' corresponding to run_name in the previous step, this action will return a folder of numpy files for all video in test list.

After the prediction step, the frame-wise Acc is measured in 'online_local.ipynb', remember to read and set carefully every variable in these 2 files when you run.

draw_cfm.ipynb and visualize.ipynb also contain helpful functions.





