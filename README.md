# improved_DDNet

## installation
```sh
# Clone the repository
git clone https://github.com/duclvQ/improved_DDNet.git
# Navigate to the project directory
cd your-repository
pip -r install requirements.txt
```
## download train/test.pickle from [`here`](https://drive.google.com/file/d/1ymfLTFaUOsoRWN51iomPHei5vObc-5mM/view?usp=sharing) to folder "data", and the "pose_new_v2" from [`here`](https://drive.google.com/file/d/1E8oAt4OI9zKblwNON-o7Wts9FfxrcWRR/view?usp=sharing):
# Location.
./data/COBOT/train.pickle     
./data/COBOT/test.pickle   
./pose_new_v2

## Training
```sh
python train.py  --run_name {run name} #  find the list of "run_name" in "models/DDNet_Original.py" 
```
## Evaluation
find pretrained weights at this ['link'](https://drive.google.com/drive/folders/1zZFcmdyBLaPQJrcfiI2OjW3QH0qAar5B?usp=sharing).
## Online test
Visit 'online.py' then set the value of 'exp' corresponding to run_name in the previous step, this action will return a folder of numpy files for all videos in test list.

After the prediction step, the frame-wise Acc is measured in 'online_local.ipynb', remember to read and set carefully every variable in these 2 files when you run.

draw_cfm.ipynb and visualize.ipynb also contain helpful functions.





