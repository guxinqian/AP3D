## [Appearance-Preserving 3D Convolution for Video-based Person Re-identification](http://arxiv.org/abs/2007.08434)

#### Requirements: Python=3.6 and Pytorch=1.0.0



### Training and test

  ```Shell
  # For MARS
  python train.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0,1 --save_dir log-mars-ap3d #
  python test-all.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d
  
  ```


### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @inproceedings{gu2020AP3D,
      title={Appearance-Preserving 3D Convolution for Video-based Person Re-identification},
      author={Gu, Xinqian and Chang, Hong and Ma, Bingpeng and Zhang, Hongkai and Chen, Xilin},
      booktitle={ECCV},
      year={2020},
    }
