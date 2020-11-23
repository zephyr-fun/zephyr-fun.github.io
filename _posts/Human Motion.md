# Human Motion Prediction via Spatio-Temporal Inpainting
---
## Abstract
**GAN**
>that have shown promising results , they can only forecast plausible motion over relatively **short** periods of time (few hundred milliseconds) and typically **ignore** the absolute position of the skeleton w.r.t. the camera.

**Proposed Methods**
>long term predictions (two seconds or more) for both the body pose and its absolute position

Three main contributions
* represent the data using a **spatio-temporal  tensor**  of  3D  skeleton  coordinates  which  allows formulating the prediction problem as an inpainting one
* design an architecture to learn the **joint distribution** of body poses and global motion, capable to hypothesize large chunks of the input 3D tensor with missing data
* argue that the **L2 metric**, considered so far by most approaches,fails to capture the actual distribution of long-term human motion(propose **two alternative metrics** , based on the distribution of frequencies , that are able to capture more realistic motion patterns.)

**while also handling situations in which past observations are corrupted by occlusions, noise and missing frames**

---
## Introduction
**motion capture technologies & large scale datasets(Human3.6M)**
**state of the art approaches**
* sequence generation task (RNNs)
* sequence-to-sequence models
* encoder-decoder predictions

**Three limitations**
* global body positioning is disregarded
* current methods require additional supervision in terms of action labels during training and inference, which limits  their  generalization  capabilities
* most approaches aim  to minimize  the L2  distance between  the ground truth and generated motion (specially to compare long motion sequence , converge to a static mean pose)

>Interestingly , the **L2 loss** is only enforced over the **reconstructed past observations** and not over the hypothesized future predictions . This way , the **generation of future frames** is fully controlled by the **discriminators**. 

>model does not require **ground  truth annotations** of the generated frames nor **explicit information** about the action being performed

>Instead of seeking to get a perfect match for alljoints across all frames (as done when using the L2 distance) , the metric we propose aims to estimate the **similarity** between **distributions** over the human motion manifold

**Sequence completion** and **image inpainting** are very related problems
* GAN
* DAE
* VAE

**Metrics for Evaluating**
* use deep metric learning and a contrastive loss to learn the metric  directly  from  the  data
* does not require training , would be to use frequency based metrics(power spectrum based metric compares sequence by sequence)
* we propose new metrics based on the distribution of the frequencies of the generated samples

The Frechet Inception Distance propose instead to fit two multivariate Gaussians to the activations of the inceptionnetwork , when the real and generated samples respectively . Then the FID is obtained by measuring the distance between the gaussian models
![model](C:\Users\16208\Desktop\WorkSpace\Done\batch_7\image\Human_flow.png)

---
## Problem Formulation
**For human pose representation**
they represent human pose with a **J-joint skeleton**, where each joint consists of its **3D Cartesian coordinates** expressed in the camera reference frame
**For motion**
A motion sequence is a concatenation of F skeletons, which we shall represent by a tensor $$S\in R^{F×J×3}$$
**For mask**
define an **occlusion mask** as a binary matrix $$M\in B^{F×J×3}$$ , which determines the part of the sequence that is not observed and is applied onto the sequence by performing the element-wise dot product $$S◦M≡S^m$$

**Note that depending on the pattern used to generate the occlusion maskM, we can define different sub-problems**
* masking the **last frames** represent a **forecasting problem**
* mask **specific intermediate joints** represents **random joints occlusions , structured occlusions or missing frames**

---
## Model
**STMI-GAN (Spatio-Temporal Motion In-painting GAN)**

* Generator
	* ($S^m$ cannot be directly processedby a convolutional network because the dimension corresponding to the joints ($J$) does not have a spatial meaning , in contrast to the temporal ($F$) and Cartesian coordinates dimensions) the generator is placed in-between a frame autoencoder, namely a frame encoder $Φ^e$ and a frame decoder $Φ^d$
	* operates over the J-dimension and projects each frame $S^m_i::\in R^{J×3}$ of the sequence to a one-dimensional vector $S^e_i::=Φ^e(S^m_i::)\in R^{H×1}$ , where $H$ is the dimension of the space for the pose embedding , denote the encoded sequence as $S^e\in R^{F×H×1}$
	* This encoded sequence is then passed through a series of generator blocks$Φ^g$, which produces a new sequence $S^g\in R^{F×H×1}$ in the embedded space
	* Finally the decoder network $Φ^d$ maps back the the transformed sequence $S^g$ to the output sequence in the original shape $S^{out}\in R^{F×J×3}$
* Discriminator
	* **Base discriminator** The same architecture of the frame encoder $Φ^e$ , with independent parameters , is used to process the generated sequence $S^{out}$
	* **EDM discriminator**  evaluates the anthropomorphism of the generated sequence $S^{out}$ via the analysis of its Euclidean Distance Matrix , computed as $EDM(S^{out})≡D\in R^{J×J}$ , where $D_{ij}$ is the Euclidean distance between joints $i$ and $j$ of $S^{out}$ , allowing to focus the attention of the discriminator into the shape of the skeleton
	* **Motion discriminator** operates over the concatenation of both , the temporal differences of absolute coordinates $‖S^{out}(t)−S^{out}(t−1)‖_1$ and the temporal differences of EDM representations $‖EDM(S^{out}(t))−EDM(S^{out}(t−1))‖_1$ , where $S^{out}(t)$ indicates generated the sequence at time $t$
* Loss
	* Reconstruction Loss
	$L_{rec}=‖(S^{out}◦M)−(S◦M)‖_2$
	* Limb Distances Loss
	$L_{limb}=\sum^F_{f=1}\sum_{\{i,j\}\in E}‖‖S^m_{fi:}−S^m_{fj:}‖_2−‖S^{m,out}_{fi:}−S^{m,out}_{fj:}‖_2‖_2$
	* Bone Length Loss
	$L_{bone}=\sum^F_{f=1}\sum^B_{b=1}‖\overline{l}_b−l_{fb}‖_2$
	* Regularized Adversarial Loss
		* Discriminator Loss
		$L_D=ESout∼Po[log(1−Dψ(Gθ(S◦M)))](4)+ES∼Po[log(Dψ(x))] +γ2ES∼Po[‖∇Dψ(x)‖2]$
		* Generator Loss
		$LG(θ,ψ) =ESout∼Po[log(Dψ(Gθ(S◦M)))]$
	* Full Loss
	$L=λ_rL_{rec}+λ_lL_{limb}+λ_bL_{bone}+λ_DL_D+λ_GL_G$
	$G^*= arg min_Gmax_{D\in D}L$

---
## Metrics for motion prediction
The **power spectrum** of a feature is computed as : 

$$PS(s_f) =‖FFT(s_f)‖^2$$

Then compute the **Power Spectrum Entropy** over a dataset : 

$$PSEnt(D) =\frac{1}{S}\sum_{s\in D}\frac{1}{F}\sum^F_{f=1}−\sum^E_{e=1}‖PS(s_f)‖∗log(‖PS(s_f)‖)$$

where $D$ is a dataset , $s$ is a sequence , $f$ is a feature, and $e$ is frequency
**PSKL** measures the distance (in terms of the KL divergence) between the ground truth and generated datasets : 

$$PSKL(C,D) =\sum^E_{e=1}‖PS(C)‖∗log(‖PS(C)‖‖PS(D)‖)$$

---
## Implementation Details
![model](C:\Users\16208\Desktop\WorkSpace\Done\batch_7\image\Human_component.png)


![model](C:\Users\16208\Desktop\WorkSpace\Done\batch_7\image\Human_3_ways.png)