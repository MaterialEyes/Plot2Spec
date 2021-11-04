# Plot2Spectra: an Automatic Spectra Extraction Tool
[manuscript available at](https://arxiv.org/abs/2107.02827)

## Abstract
Different types of spectroscopies, such as X-ray absorption near edge structure (XANES) and Raman spectroscopy, play a very important role in analyzing the characteristics of different materials. In scientific literature, XANES/Raman data are usually plotted in line graphs which is a visually appropriate way to represent the information when the end-user is a human reader. However, such graphs are not conducive to direct programmatic analysis due to the lack of automatic tools. In this paper, we develop a plot digitizer, named Plot2Spectra, to extract data points from spectroscopy graph images in an automatic fashion, which makes it possible for large scale data acquisition and analysis. Specifically, the plot digitizer is a two-stage framework. In the first, the axis alignment stage, we adopt an anchor-free detector to detect the plot region and then refine the detected bounding boxes with an edge-based constraint to locate the position of two axes. We also apply scene text detector to extract and interpret all tick information below the x-axis. In the second, the plot data extraction stage, we first employ semantic segmentation to separate pixels belonging to plot lines from the background, and from there, incorporate optical flow constraints to the plot line pixels to assign them to the appropriate line (data instance) they encode. Extensive experiments are conducted to validate the effectiveness of the proposed plot digitizer, which shows that such a tool could help accelerate the discovery and machine learning of materials properties.


## 🤔 Consider Collaboration

If you find this tool or any of its derived capabilities useful, please consider registering as a user of Center for Nanoscale Materials. We will keep you posted of latest developments, as well as opportunities for computational resources, relevant data, and collaboration. Please contact Maria Chan (mchan@anl.gov) for details.




## v1.0

### set up environment
```bash
conda env create -f Plot2Spec.yml
``` 
- Python 3.8.5
- PyTorch 1.7.0
- CUDA 11.0 
- GCC 7.3
- mmdet 2.11.0

mmdet may be installed via https://github.com/open-mmlab/mmdetection

### try demo script (demo.ipynb)

- weights can be downloaded from Google Drive https://drive.google.com/drive/folders/1zELrF8LzKVCGdH1QI5bCfNcDbvkC2sFp?usp=sharing

put the checkpoints.zip in the root dir and unzip it.
```bash
unzip checkpoints.zip
``` 
 

### TODO: 
- [ ] training code
- [ ] automatic parameter selection


## Acknowledgements <a name="credits"></a>
This material is based, in part, upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357. M.C. acknowledges the support from the BES SUFD Early Career award. Use of the Center for Nanoscale Materials, an Office of Science user facility, was supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, under Contract No. DE-AC02-06CH11357. We gratefully acknowledge the computing resources provided and operated by the Joint Laboratory for System Evaluation (JLSE) at Argonne National Laboratory.


## Citation
If you find Plot2Spectral useful, please encourage its development by citing the following [paper](https://arxiv.org/abs/2107.02827) in your research:
```
Jiang, Weixin, Eric Schwenker, Trevor Spreadbury, Kai Li, Maria KY Chan, and Oliver Cossairt. "Plot2Spectra: an Automatic Spectra Extraction Tool." arXiv preprint arXiv:2107.02827 (2021).
```

#### Bibtex
```
@article{jiang2021plot2spectra,
  title={Plot2Spectra: an Automatic Spectra Extraction Tool},
  author={Jiang, Weixin and Schwenker, Eric and Spreadbury, Trevor and Li, Kai and Chan, Maria KY and Cossairt, Oliver},
  journal={arXiv preprint arXiv:2107.02827},
  year={2021}
}
```