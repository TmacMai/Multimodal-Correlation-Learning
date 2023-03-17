# Multimodal-Correlation-Learning

Here is the code for the paper "Excavating Multimodal Correlation for Representation Learning". The paper is accepted by Information Fusion.

Firstly, we would like to express our gratitude to the authors of MAG-BERT (https://github.com/WasifurRahman/BERT_multimodal_transformer). Their codes are of great help to our research.

To run the code, you firstly need to download the data using (see https://github.com/WasifurRahman/BERT_multimodal_transformer for more details):

     sh datasets/download_datasets.sh

We have already provided the processed mosi data. For the larger mosei dataset, you should download it by the above command.

Then install the required packages using:

     pip install -r requirements.txt

Finally, we can run the codes using the following command:

To run the MCL (joint training):

     python main_mcl.py --dataset mosi --train_batch_size 35 --n_epochs 50


Currently two datasets are provided, i.e., mosi and mosei (please refer to https://github.com/A2Zadeh/CMU-MultimodalSDK for more details about the datasets). To run with mosei dataset, you should firstly open the global_configs.py, and then change the VISUAL_DIM to 47. We download the datasets by running (https://github.com/WasifurRahman/BERT_multimodal_transformer):

    pip install gdown

    gdown https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou

    gdown https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3.

Finally, we can run the code using the following commands (change the mib parameter to your desired one):

     python main_mcl.py --dataset mosei --train_batch_size 40 --n_epochs 50 --gamma 1 --d_l 50
     
We will continue to update the codes.

Notbly, following https://github.com/WasifurRahman/BERT_multimodal_transformer, when calculating corr and MAE, we do not use the neutral utterances.

If you find our code useful, please cite our paper:

@article{mai2023excavating,

  title={Excavating multimodal correlation for representation learning},
  
  author={Mai, Sijie and Sun, Ya and Zeng, Ying and Hu, Haifeng},
  
  journal={Information Fusion},
  
  volume={91},
  
  pages={542--555},
  
  year={2023},
  
  publisher={Elsevier}
}
