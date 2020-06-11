
# Attention Sampling - Pytorch
This is a PyTorch implementation of the the paper: ["Processing Megapixel Images with Deep Attention-Sampling Models"](https://arxiv.org/abs/1905.03711). This repository is based on the [original repository](https://github.com/idiap/attention-sampling) belonging to this paper which is written in TensorFlow.

## Porting to PyTorch
The code from the original repository has been rewritten to to a PyTorch 1.4.0 implementation. The most difficult part was rewriting the functions that extract the patches from the high resolution image. The original version uses special C/C++ files for this, I have done this in native Python. This is probably more inefficient and slower because it requires a nested for-loop. I tested with performing the patch extraction in parallel but this adds so much overhead that it is actually slower. 

Furthermore, I hope I implemented the part where the expectation is calculated correctly. This uses a custom `backward()` function and I hope there are no bugs in it. 

##  Performance
This code repository has been tested on two of the tasks mentioned in the original paper: the Mega-MNIST and the traffic sign detection task. This code base yields comparable results on both tasks. However, the traffic sign detection task requires approximately twice as many epochs to achieve similar results. These experiments can be run by running `mega_mnist.py` and `speed_limits.py`.

## Installation
Dependencies can be found inside the `requirements.txt` file. To install, run `pip3 install -r requirements.txt`. This code repository defaults to running on a GPU if it is available. It has been tested on both CPU and GPU.

## Questions and contributions
If you have any question about the code or methods used in this repository you can reach out to joris.mollinga@surfsara.nl. If you find bugs in this code (which could be possible) please also contact me or file an issue. If you want to contribute to this code my making it more efficient (for example, the patch extraction procedure is quite inefficient) please contact me or submit a pull request. 

## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 
![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">



