This project is a cleaned up version of our PAMI submission 
"Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly" in tensorflow. It extends our
original ICCV version with an adversarial auxiliary loss during training, which improves results. 
you are planning to use this work, please cite one of the following papers:


    @inproceedings{opitz2017iccv,
        author={M. Opitz and G. Waltner and H. Possegger and H. Bischof},
        booktitle={{ICCV}},
        title={{BIER: Boosting Independent Embeddings Robustly}},
        year={2017}
    }

    @article{opitz2018arxiv,
        author={M. Opitz and G. Waltner and H. Possegger and H. Bischof},
        journal={arXiv:cs/1801.04815},
        title={{Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly}},
        year={2018}
    }

To run the code, see ./run.sh and ./run_eval.sh.
The train-images file is a numpy file consisting of images of size 256x256 and train-labels are the corresponding
labels. The label indices should be between [0, total-number-of-labels) (i.e. they should be non-negative, and continuous).
