pip install -r requirements.txt
mkdir datamount
cd datamount
kaggle datasets download -d iafoss/stanford-ribonanza-rna-folding-converted
unzip -n stanford-ribonanza-rna-folding-converted.zip
rm stanford-ribonanza-rna-folding-converted.zip
cd ..
cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/repos/EternaFold/src
make contrafold
cd /workspace/Stanford-Ribonanza-RNA-Folding-32th-place-solution/
python summary.py
python train.py -C cfg_0
python infer.py --Checkpoint ['datamount/weights/cfg_0/fold0/checkpoint_last_seed2023.pth']
