LSGANs with Gradient Regularizers are Smooth High-dimensional Interpolators
====================

## Introduction

This is the code accompanying the NeurIPS 2022 "INTERPOLATE: First Workshop on Interpolation Regularizers and Beyond" Workshop paper on "LSGANs with Gradient Regularizers are Smooth High-dimensional Interpolators" All codes are in Tensorflow2.0 Keras, and can be implemented on either GPU or CPU.

## Poly-LSGAN RBF

The central component of Poly-WGANs is the radial basis function implementation, and can be found in the ``gan_topics.py`` file, defined under the ``RBFSolver`` and ``PHSLayer`` classes. A reviewer interested in getting to the core of these methods is requested to check this part of the code to get insights on the RBF disciminator implementation, weight and centre computation, lambda_d computation, etc. 

## Poly-LSGAN RBF Network Architecture

A slice of the RBF Network code is given below. In TF, the code for building the RBF would look like this:

```
def discriminator_model_RBF(self):

    if self.data not in ['g2','gmm8']:
        inputs = tf.keras.Input(shape=(self.output_size,self.output_size,self.output_dims))
        inputs_res = tf.keras.layers.Reshape(target_shape = [self.output_size*self.output_size*self.output_dims])(inputs)
    else:
        inputs = tf.keras.Input(shape=(self.latent_dims,))
        inputs_res = inputs

    num_centers = 2*self.batch_size

    if self.gan == 'LSGAN':
        [Out,A,B] = PHSLayer(num_centres=num_centers, output_dim=1,  dim_v = self.dim_v, rbf_k = self.rbf_m, batch_size = self.batch_size, multi_index = self.multi_index)(inputs_res)
        model = tf.keras.Model(inputs=inputs, outputs= [Out,A,B])
    

    return model
```

The code for matrix inversion for finding the weights of the PHS interpolation can be found within the same class, and in given below for reference:

```
def PHS_MatrixSolver(self, vals):#, centers, vals, phs_deg, poly_deg):
    N = self.A.shape[0]
    # print(self.B)
    Correction = -96*3.14159*self.lambda_D*tf.eye(N) ### Constants from Aronjszan et al, 1893 (Check Main Manuscript)

    self.BT = tf.transpose(self.B)
    M = tf.concat( (tf.concat((self.A+Correction,self.B), axis = 1),tf.concat((self.BT,self.LZeroMat), axis = 1)), axis = 0)

    y = tf.concat((vals,tf.zeros((self.dim_v,1))), axis = 0)
    sols = tf.linalg.solve(M,y)

    Weights = tf.squeeze(sols[0:N])
    PolyWts = sols[N:]
    
    return Weights, PolyWts
```

## Poly-LSGAN Anaconda Environment


This codebase consists of the TensorFlow2.x implementation Poly-WGAN and PolyGAN-WAE. The baseline comparisons are with multiple WGAN, GMMN and WAE variants as described in the paper. All results were optained when training on TF2.5. Other version of TF might cause instability due to library deprications. Use with care. 

Dependencies can be installed via anaconda. The ``PolyGAN_GPU_TF25.yml`` file list the dependencies to setup the GPU system based environment. To install from the yml file, please run ``conda env create --file PolyGAN_GPU_TF25.yml`` : 

```
GPU accelerated TensorFlow2.0 Environment:
dependencies:
  - cudatoolkit=11.3.1
  - cudnn=8.2.1=cuda11.3_0
  - opencv=3.4.2
  - pip=20.0.2
  - python=3.6.10
  - pytorch=1.4.0=py3.6_cpu_0
  - pip:
    - absl-py==0.9.0
    - clean-fid==0.1.15
    - h5py==2.10.0
    - matplotlib==3.1.3
    - numpy==1.19.5
    - pot==0.8.0
    - tensorflow-addons==0.13.0
    - tensorflow-datasets==4.4.0
    - tensorflow-estimator==2.5.0
    - tensorflow-gpu==2.5.0
    - tensorflow-probability==0.13.0
    - torch==1.10.0+cu113
    - torchaudio==0.10.0+cu113
    - torchvision==0.11.1+cu113
    - tqdm==4.42.1
```
If a GPU is unavailable, the CPU only environment can be built  with ``PolyGAN_CPU.yml``. This setting is meant to run evaluation code;. Training on the CPU environment is not advisable.

```
CPU based TensorFlow2.0 Environment:
- pip=20.0.2
- python=3.6.10
- opencv=3.4.2
- pytorch=1.4.0=py3.6_cpu_0
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - clean-fid==0.1.15
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - pot==0.8.0
    - tensorboard==2.0.2
    - tensorflow-addons==0.6.0
    - tensorflow-datasets==3.0.1
    - tensorflow-estimator==2.0.1
    - tensorflow==2.0.0
    - tensorflow-probability==0.8.0
    - tqdm==4.42.1
    - gdown==3.12
```

If a Conda environment with the required dependecies already exists, or you wish to use your own environment for any particular reason, we suggest making a clone called PolyGAN to maintain consistent code-running experiemnt with the exisitng bash files: ``conda create --name PolyGAN --clone <Your_Env_Name>``

The ``clean-fid`` and ``pot`` libraries are used for metric computation (cf. Appendix E).

Codes were tested locally and on Google Colab. Implementation Jupyter Notebooks to recrete the results shown in the paper will be included with the camera-ready version of the paper. The current version of the code allows implementing locally, the training codes for the various models considered.

In case of runtime issues occuring due to hardware/driver incompatibitlity, please refer the associated user-manuals of NVIDIA CUDA, CudNN, PyTorch or TensorFlow to install dependecies from source.


## Training Data

MNIST and Fashion-MNIST are loaded from TensorFlow-Datasets. The CelebA datasets must be downloaded by running the following code:
```
python download_celeba.py
```


## Training  

The code provides training procedures on MNIST, Fashion-MNIST, and CelebA datasets. We include training codes for all basic experinets presented in the paper.

1) **Running ``train_*.sh`` bash files**: There are two bash files to run an example instacne of trainging: ``train_PolyGAN_GPU.sh`` and ``train_Baseline_GPU.sh``. The fastest was to train a model is by running these bash files. Around 3-4 sample functions calls are provided in the above bash files.  Uncomment the desired command to train for the associated testcase.
```
bash bash_files/train/train_PolyGAN_GPU.sh
bash bash_files/train/train_Baseline_GPU.sh
```
(While code is provided for running experiments on GPU, their CPU counterparts can be made by seting the appropriate flags, as discussed in the next section.)
2) **Manually running ``gan_main.py``**: Aternatively, you can train any model of your choice by running ``gan_main.py`` with custom flags and modifiers. The list of flags and their defauly values are are defined in  ``gan_main.py``.    

## Flags

The default values of the various flags used in the bash files are discussed in ``gan_main.py``. While most flags pertain to training parameters, a few control the actula GAN trining algorithm:

1) ``--gan``: Can be set to ``LSGAN`` based on which set of experiments are to be executed. The corresponding file from ``models/GANs/`` will be imported. Consider adding your own variants if required.
2) ``--topic``: The kind of GAN varinat to consider. ``Base`` calls the class associated with trainig baselines WGAN variants (GP, LP, Rd (or R1 as the oirignal authors call it), Rg (or R2), DRAGAN and and ALP. ``PolyGAN`` results in calling those classes associated with PolyGAN training. 
3) ``--loss``: Choice of GAN loss. For WGAN, there is ``base``,``GP``, ``LP``,``ALP``,``DRAGAN``, ``R1`` and ``R2`` available for training. The proposed loss variant is ``RBF``.
4) ``--mode``: Choose between ``train``, ``test``, or ``metrics``. While ``test`` genetares interpolation and reconstruction results, the ``metrics`` mode allows for evaluating FID, sharpness, etc. 
5) ``--metrics``: A comma seperated string of metrics to compute. Currently, it supports ``FID,sharpness,W22,GradGrid``. When passing multiple, please avoid blank spaces.
6) ``--data``: Target data to train on. Support ``mnist``,``fmnist``,and ``celeba``. (All lowercase only)
7) ``--latent_dims``: The letent space dimensionality to use in WAE variants. for 2D and nD Gaussians, this is atuomatically assumed to be the dimensionality of the data itself.
8) ``--rbf_m``: The gradient pentaty order, brought to life via the order of the polyharmonic RBF order 2m-n. Optimal performance when m = ceil(n/2)
9) ``--GPU``: A list indicating the visble GPU devices to CUDA. Set to ``0``, ``0,1``, ``<empyt>``, etc. based on compute available.
10) ``--device``: The device that TensorFlow places its variables on. Begins iteration from ``0`` all the way up to ``n``, when `n` devies are proved in ``--GPU``. Set ``--GPU`` as empty string and ``--device`` to ``-1`` to train on CPU. 



----------------------------------
----------------------------------

----------------------------------

**Siddarth Asokan**  
**Ph.D. Scholar, RBCCPS**
**Indian Institute of Science**
**Bangalore, India**
**EMAIL: siddartha@iisc.ac.in, siddarth.asokan@gmail.com**  

----------------------------------
----------------------------------