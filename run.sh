# conda env create -f environment.yaml
# conda activate mshnet
python main.py --dataset-dir './datasets/IRSTD-1k' --batch-size 64 --epochs 400 --lr 0.05 --mode train