# Social Spatio-Temporal Graph Convolutional Neural Network
---
## Abstract
topic layer one : 
```
understanding of pedestrian behaviors
```
topic layer two :
```
modeling interactions betweenagents such as autonomous vehicles and humans
```
trajectories are influenced by : 
* itself (self aim)
* interaction with surrounding objects
**previous methods of modeling interactions :** 
```
aggregates the learned peedstrains states
```
paper proposes the **Social STGCNN method** to substitute aggregation methods by modeling the interactions as a graph .

**results**
* FED **20%** improvement over the state of the art method
* ADE **8.5** times less parameters and up to **48** times faster
* data efficient, and exceeds previous state of the art on the ADE metric with only **20%** of the training data
* a **kernel function** to embed the **interactions** within the **adjacency matrix**
code is available at https://github.com/abduallahmohamed/Social-STGCNN

---
## Introduction
The trajectory of a pedestrian is **challenging to predict** , due to the **complex interactions** between the **pedestrian** with the **environment** , include :
* **physical obstacles** such as trees or roads 
* **moving objects** including vehicles and other pedestrians 
> 70% of pedestrians tend to walk in groups.

**interactions between** pedestrians are mainly driven by : 
* common sense 
* social conventions
**complexity** of pedestrian trajectory prediction : 
* different **social behaviors** such as walking in parallel with others, within a group, collision avoidance and merging from different directions into a specific point
* **randomness of the motion**(given that the target destination and intended path of the pedestrian are unknown.)

**Related work**
* **Social-LSTM**
> modeling each pedestrian trajectory via a **recurrent deep model** , the outputs of recurrent models are made to **interact** with each other via a **pooling layer** , view pedestrian trajectories as a **bi-variate Gaussian distribution**

* **GAN**

> assuming that the distribution of trajectories is **multi-modal** . generators are designed using recurrent neural networks, and again, **aggregation methods** are relied upon to **extract the social interactions** between pedestrians.

**Limitation of earlier works comes from : **

* the use of RNN architectures
* aggregation layers
	* the aggregation in feature states is neither intuitive nor direct in modelling inter-actions between people, as the physical meaning of feature states is difficult to interpret
	* aggregation mechanisms are usually based on heuristics like pooling,they could fail in modeling interactions between pedestrians correctly

**The paper uses TCN and Graph(to represent interactions between pedestrians) instead **
**Design of the proposed model**
* spatio-temporal graph to represent the interactions(edges->weighted->kernel function)
* address the RNN issues : using a graph Convolutional Neural Network(GCN) and a TCN

**prediction accuracy , parameters size , inference speed , data efficiency**

---
## Problem Formulation
a set of $$N$$ pedestrians with their observed positions  $$ tr_o^n , n\in \{1,...,N\} $$ over a time period $$ T_o $$ , to predict the upcoming trajectories $$tr_p^n$$ over a future time $$ T_p $$

trained to minimize the **negative log-likelihood** , which defined as:

>$$L^n(W) =-\sum^{T_p}_{t=1}{ log(P((p^n_t|\hatμ^n_t,\hatσ^n_t,\hatρ^n_t))}$$

in which $$ W $$ includes all the **trainable parameters** of the model , $$μ^n_t$$ is the **mean** of the distribution , $$σ^n_t$$ is the **variances** and $$ρ^n_t$$ is the **correlation**

---
## Model Description
**Social-STGCNN** model consists of **two** main parts : 
* the Spatio-Temporal Graph Convolution Neural Network(ST-GCNN)
>the ST-GCNN conducts spatio-temporal convolution operations on the graph representation of pedestrian trajectories to extract features . These features are a compact representation of the observed pedestrian trajectory history

* the Time-Extrapolator Convolution Neural Network(TXP-CNN)
>TXP-CNN takes these features as inputs and predicts the future trajectories of all pedestrians as a whole

**graph representation of pedestrian trajectories**
$$ G_t $$ represents the relative locations of pedestrians in a scene at each time step $t$
defined as $$ G_t=(V_t , E_t) $$ , where $$V_t=\{v^i_t|i\in \{1 , ... , N\}\}$$ , The observed location$$(x^i_t,y^i_t)$$ is the attribute of $$v^i_t$$ . $$E_t$$ is the set of edges within graph$$G_t$$ which is expressed as $$E_t=\{e^{ij}_t| ∀i , j\in \{1 , ... , N\}\} . $$ $$e^{ij}_t= 1$$ if $$v^i_t$$ and $$v^j_t$$ are connected , $$e^{ij}_t= 0$$ otherwise.
In order to model how strongly two nodes could influence with each other, we attach a value $$a^{ij}_t$$ ,  which is computed by some **kernel function** for each $$e^{ij}_t$$
$$
a^{ij}_{sim,t}=\begin{cases} \frac{1}{||V^i_t - V^j_t||_2} , ||V^i_t - V^j_t||_2 \neq 0 \\ 0 , Otherwise  \end{cases}
$$

**GCN**
$$
v^{i(l+1)}= \sigma( \frac{1}{\Omega} \sum_{v^{j(l)}\in B(v^{i(l)})}{p(v^{i(l)} , v^{j(l)}).w(v^{i(l)} , v^{j(l)})})
$$
where $$\frac{1}{\Omega}$$ is a normalization term , $$B(v^i) ={v^j|d(v^i,v^j)≤D}$$ is the neighbor set of vertices $v^i$ and $d(v^i,v^j)$ denotes the shortest path connecting $v^i$ and $v^j$ .  Note that $\Omega$ is the cardinality of the neighbor set.
**Spatio-Temporal Graph Convolution Neural Network(ST-GCNN)**
>The functionality of ST-GCNN is to extract spatio-temporal node embedding from the input graph , We denote the embedding resulting from ST-GCNN as$\overline{V}$

**Time-Extrapolator Convolution Neural Network(TXP-CNN)**
>TXP-CNN operates directly on the temporal dimension of the graph embedding$\overline{V}$ and expands it as a necessity for prediction

**Whole Flow**

![whole flow](C:\Users\16208\Desktop\WorkSpace\Done\batch_7\image\whole_flow.png)
**two main differences between Social-STGCNN and ST-GCNN**
* Social-STGCNN constructs the graph in a totally different way from ST-GCNN with a novel kernel function.
* beyond the spatio-temporal graph convolution layers, we added the flexibility in manipulating the time dimension using the TXP-CNN . ST-GCNN was originally designed for classification

---
## Implementing
* normalize the adjacency matrix
> $$ A_t = D^{-\frac{1}{2}}_t\hat{A}_tD^{-\frac{1}{2}}_t $$

* ST-GCNN
> $$ f(V^{(l)} , A) = \sigma(D^{-\frac{1}{2}}_t \hat{A}_tD^{-\frac{1}{2}}_tV^{(l)}W^{(l)})$$

* TXP-CNN
> the TXP-CNN receives features $$ \overline{V} $$ and treats the time dimension as feature channels

---
## Datasets and Evaluation Metrics
**Datasets : ** 
* **ETH**(contains two scenes)
	* ETH
	* HOTEL
* **UCY**(contains three scenes) 
	* ZARA1
	* ZARA2
	* UNIV

**The trajectories in datasets are sampled every 0.4 seconds** , When being evaluated, the model **observes** the trajectory of **3.2** seconds which corresponds to **8** frames and predicts the trajectories for the next **4.8** seconds that are **12** frames . Intuitively , **ADE** measures the average prediction performance along the trajectory,  while the **FDE** considers only the prediction precision at the end points

Since Social-STGCNN generates a **bi-variate Gaussian distribution** as the prediction, to compare a **distribution** with a **certain target value** , we follow the evaluation method used in Social-LSTM in which **20 samples** are generated based on the predicted distribution.  Then the ADE and FDE are computed using the closest sample to the ground truth

**it is noticeable that when the number of ST-GCNN layers increases, the model performance decreases**
![visual](C:\Users\16208\Desktop\WorkSpace\Done\batch_7\image\visual.png)
