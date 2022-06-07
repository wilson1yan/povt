# Patch-based Object-centric Transformers for Efficient Video Generation
[[Paper]]()[[Website]](https://sites.google.com/view/povt-public)

In this work, we present Patch-based Object-centric Video Transformer (POVT), a novel region-based video generation architecture that leverages object-centric information to efficiently model temporal dynamics in videos. We build upon prior work in video prediction via an autoregressive transformer over the discrete latent space of compressed videos, with an added modification to model object-centric information via bounding boxes. Due to better compressibility of object-centric representations, we can improve training efficiency  by allowing the model to only access object information for longer horizon temporal information. When evaluated on various difficult object-centric datasets, our method achieves better or equal performance to other video generation models, while remaining computationally more efficient and scalable. In addition, we show that our method is able to perform object-centric controllability through bounding box manipulation.

# Installation
```
conda create -n povt python=3.7
conda activate povt
conda install --yes pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install -e .
```

# Datasets
Example dataset video can be visualized with `scripts/visualize_dataset.py`

## CATER-CAM
Download CATER (with masks) from the [Deepmind Multi-Object Dataset](https://github.com/deepmind/multi_object_datasets). 

To convert it to HDF5 format, run `python data_gen/cater_cam/cater_cam_to_hdf5.py /PATH/TO/DATASET`

Then, to compute bounding boxes, run `python data_gen/cater_cam/compute_cater_cam_bboxes.py -d data/cater_cam.hdf5`

## CATER-MOV
We [use a modified version of the original CATER repo](https://github.com/wilson1yan/CATER) to render segmentation masks. You will need to first download blender 2.79b and install the appropriate python libraries, or use their provided singularity container. Run `cd CATER/generate` and generate samples with `python launch.py --num_jobs 4` (total number of jobs with be # of GPUS * num_jobs). You may need to set `PYTHONPATH` to the `generate` directory. The dataset will be saved to `CATER/generate/datasets`

To convert it to HDF5 format, run `python data_gen/cater_mov_to_hdf5.py -d /PATH/TO/cater_mov` to produce `cater_mov.hdf5`

## Something-Something V2
We [use a modified version of the original SORT repo](https://github.com/abewley/sort). Please follow the steps below:

1. Download Something-Something V2 found [here]() into a folder `povt/data/something-something`
2. `cd povt/data/something-something` and extract data files with `cat 20bn-something-something-v2-?? | tar zx`
3. Download each bounding box annotation file found [here](https://github.com/joaanna/something_else), and place the `.json` files into `povt/data/something-something/annotations`
4. `cd povt` and run `python data_gen/ssv2/read_bboxes.py data/something-something/`
5. `cd sort` and run `python bbox_to_txt.py /PATH/TO/something-something`.
6. You may need to first install SORT dependencies from `sort/requirements.txt`. Then run `python sort.py --seq_path /PATH/TO/something-something/`
7. Move all generated `.txt` files to replace the ones in `/PATH/TO/something-something/20bn-something-something-v2` with the command `find output/ -name '*.txt' -exec mv {} /PATH/TO/something-something/20bn-something-something-v2 \;`
8. `cd povt` and run `python data_gen/ssv2/filter.py`

# Training VQ-VAE
To train a VQ-VAE with 2 GPUs, run `CUDA_VISIBLE_DEVICES=0,1 python scripts/vqvae/run.py -o <exp_name> -c configs/base_vqvae.yaml`

Reconstructions can be visualized with `scripts/visualize_reconstructions.py`

# Training POVT
Make sure to fill `configs/base_povt.yaml` with the correct VQ-VAE checkpoint. To train POVT with 4 GPUs, run `CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt/run.py -o <exp_name> -c configs/base_povt.yaml`

Samples can be generated with `scripts/sample.py`. Note that the default code performs single-frame conditional video prediction.
