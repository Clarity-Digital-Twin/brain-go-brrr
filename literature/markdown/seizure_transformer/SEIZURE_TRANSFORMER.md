# SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings



3 [rd] B¨ulent Yener

_Department of Computer Science_
_Rensselaer Polytechnic Institute_
Troy, NY. USA.
yener@cs.rpi.edu



1 [st] Kerui Wu

_Department of Computer Science_
_Rensselaer Polytechnic Institute_
Troy, NY. USA.
wuk9@rpi.edu



2 [nd] Ziyue Zhao
_Department of Computer Science_
_Rensselaer Polytechnic Institute_
Troy, NY. USA.
zhaoz10@rpi.edu



_**Abstract**_ **—Epilepsy is a common neurological disorder that**
**affects around 65 million people worldwide. Detecting seizures**
**quickly and accurately is vital, given the prevalence and severity**
**of the associated complications. Recently, deep learning-based**
**automated seizure detection methods have emerged as solutions;**
**however, most existing methods require extensive post-processing**
**and do not effectively handle the crucial long-range patterns**
**in EEG data. In this work, we propose SeizureTransformer,**
**a simple model comprised of (i) a deep encoder comprising**
**1D convolutions (ii) a residual CNN stack and a transformer**
**encoder to embed previous output into high-level representation**
**with contextual information, and (iii) streamlined decoder which**
**converts these features into a sequence of probabilities, directly**
**indicating the presence or absence of seizures at every time step.**
**Extensive experiments on public and private EEG seizure**
**detection datasets demonstrate that our model significantly out-**
**performs existing approaches (ranked in the first place in the**
**2025 ”seizure detection challenge” organized in the International**
**Conference on Artificial Intelligence in Epilepsy and Other**
**Neurological Disorders), underscoring its potential for real-time,**
**precise seizure detection.**
_**Index Terms**_ **—Time series analysis, change point detection,**
**deep learning, transformers**


I. I NTRODUCTION


Epilepsy is a prevalent neurological disorder distinguished
by recurring seizures. Worldwide, there are approximately 65
million people with epilepsy, more than Parkinson’s disease,
Alzheimer’s disease, and Multiple Sclerosis combined. One of
the most serious complications linked to epilepsy is Sudden
Unexpected Death in Epilepsy(SUDEP), which tragically results in the deaths of around 1 in every 1000 epilepsy patients
each year [1]. Given the severity of this risk, early and precise
seizure detection is crucial in clinical practice, as prompt
intervention can considerably lower mortality rates [2].
Traditionally, large numbers of multi-channel EEG signals
are visually analyzed by neurologists with the goal of understanding when and where the seizures start and how they
propagate within the brain. However, there are two main
disadvantages of visual analysis of EEG signals: it is timeconsuming and prone to subjectivity. Therefore, automation
of the detection of the underlying brain dynamics in EEG



signals is significant in order to obtain fast and objective EEG
analysis.
EEG signals can be treated as a batch of time series, a
sequence of data points indexed in a discrete-time order, which
formulates the automated seizure detection problem to be
part of a classification task in time series analysis. In recent
years, deep learning models have demonstrated impressive
abilities to capture the intricate dependencies within time
series data, making them a powerful tool for time series
analysis over traditional statistical methods. However, most
existing work [3]–[9] implements the classification task at
a sliding window level, which involves segmenting a signal
recording into distinct windows and predicting a label for
each sample. Converting separated predictions into final event
prediction in Standardized Computer-based Organized Reporting of EEG (SCORE) standard [10] that can be used in real
life involves extensive time-consuming post-processing, which
departs existing algorithms from simultaneous detection. More
than that, existing time series analysis research often train and
evaluate models using datasets that have a small sequence
length [11], while EEG studies haven shown that long-range
input records can largely benefit accurate prediction [12].
In contrast to window-level classification models, sequenceto-sequence modeling, a type of encoder-decoder model to
map an input sequence to an output sequence, provides a
straightforward solution to avoid redundant post-processing
steps through time-step-level classification. In the filed of Natural Language Processing(NLP), Transformer-based models
have shown remarkable predictive and generative abilities [].
However, studies have shown that CNN-based models achieve
better classification ability in time series analysis compared
to RNN-based and Transformer-based models [13], [14]. This
lets the focus of scientific signal classification study be on the
U-Net [12], [15], [16], a fully convolutional encoder-decoder
network with skip connections that was originally designed
for image segmentation [17]. The drawback of such models
also stands out. Firstly, U-Net primarily operates within local
receptive fields, making it difficult for U-Net to effectively
model long-range dependencies as the input sequence length


becomes big. Beyond that, Scaling U-Net to large datasets
or high-resolution sequences requires stacking deeper layers,
which often leads to vanishing gradients, overfitting, and
massive memory and computation usage.
In this work, we propose a simple U-Net-based architecture, namely, SeizureTransformer, to solve the mentioned
challenges. The model comprises of three components (i)
a deep encoder comprising 1D convolutions (ii) a residual
CNN stack and a transformer encoder to embed previous
output into high-level representation with global contextual
information, and (iii) streamlined decoder which converts these
features into a sequence of probabilities, directly indicating the
presence or absence of seizures at every time step. The scaling
embedding components makes the model to be easily scalable
to build up the model size and to handle long-sequence signals.
Experimentally, our model achieves the consistent state-of-theart performance, efficiency, and generalization across diverse
subjects and devices in public and private EEG datasets. Our
model has ranked number one in an international competition
organized by the _International Conference on Artificial Intel-_
_ligence in Epilepsy and Other Neurological Disorders_ .


II. R ESULTS


_A. Model Overview_


We design model architecture based on the U-Net to do
end-to-end learning from raw waveforms for time-step-level
classification to achieve simultaneous seizure detection. Our
model consists of three primary modules: an encoder, a
scaling embedding component, and a decoder, as shown in
Fig. 1. Taking the continuous long-term EEG signals from
the epilepsy monitoring unit, the encoder extracts features
by recognizing patterns through one-dimensional convolution
layers. The feature vectors are further embedded by a ResCNN
stack and a Transformer encoder stack with a global attention
mechanism to generate high-level representations that capture
rich temporal dependencies. The streamlined decoder then
converts these representations into a sequence of probability,
indicating the presence or absence of seizures at every time
step. Residual connections between each encoder layer and
decoder layer are used to ease the gradient flow and to
avoid degradation problems in the deep neural network. More
details about network architecture selection are provided in the
methodology section.


_B. Model Training_


**Datasets.** We use Temple University Hospital EEG Seizure
Corpus v2.0.3(TUSZ) [18] and Siena Scalp EEG Database

[19] to form our training dataset. TUSZ is the largest public
dataset for seizure detection that has been manually annotated
with data for seizure events. The predefined training set in
TUSZ has 910 hours of recording sessions from 579 subjects
with various sampling frequencies, from 250 Hz to 1000 Hz.
The Siena Scalp EEG Database is a small dataset that contains
128 hours of recording sessions from 14 subjects with a unified
sampling rate of 512 Hz. Both datasets contain at least 19
electrodes of the international 10-20 system. We unify the



training data from two datasets by resampling signals into 256
Hz and fixing the channel sequence in order (Fig. 2a).
We combine two datasets by concatenating segmented oneminute-long time series windows together, i.e., 60 _×_ 256 =
15360 time steps per window. A 75% overlap ratio between
two consecutive windows was set as a hyperparameter during
the segmentation process to augment training examples. To
improve the model’s ability to distinguish seizure signals from
background noise, we statistically categorize training windows
into three classes: no-seizure, full-seizure, and partial-seizure,
and uniformly sample a certain number of windows from each
class to create a balanced dataset. Specifically, our training
dataset is constructed as follows:


_D_ = _D_ _ps_ _∪D_ _fs_ _[∗]_ _[∪D]_ _ns_ _[∗]_


where _D_ _ps_ contains all partial-seizure windows, _D_ _fs_ _[∗]_ [and] _[ D]_ _ns_ _[∗]_
is a randomly selected subset of full-seizure and no-seizure
window with _|D_ _fs_ _[∗]_ _[|]_ [ = 0] _[.]_ [3] _[ × |D]_ _[ps]_ _[|]_ [ and] _[ |D]_ _ns_ _[∗]_ _[|]_ [ = 2] _[.]_ [5] _[ × |D]_ _[ps]_ _[|]_ [.]
**Pre-processing.** We followed [6]’s process for preprocessing
EEG data before feeding into the model using a bandpass filter
to keep frequencies in a range from 0.5 Hz to 120 Hz and two
notch filters to eliminate signals at 1 Hz and 60 Hz, which are
typically associated with heart rate and power line noise (Fig.
2b).
**Training Setting.** We implemented our deep learning model
using PyTorch and trained on 2 parallel NVIDIA L40S 46GB
GPUs. Our training parameters include a batch size of 256,
a learning rate of 1e-3, a weight decay of 2e-5, and a drop
rate of 0.1 for all dropout layers both at training and test time.
We use Binary Cross-Entropy loss as the objective function
and RAdam as the optimizer. The training process was set
to be 100 epochs with early stopping if no improvement in
validation loss was observed over 12 epochs.
**Post-processing.** After having a sequence of probabilities,
outputted by the model, we implement a set of simple postprocessing steps to convert continuous probabilities to the
final detection(Fig. 2c). Initially, we apply a straightforward
threshold filter to obtain a discrete mask. Then, two morphological operations are employed to eliminate spurious spikes of
seizure activity and to fill short 0 gaps. Lastly, we implement a
simple duration-based rule to discard blocks of seizure labels
lasting less than a minimal clinically relevant duration.


_C. Evaluation Results_


We used TUSZ’s predefined test set, consisting of 42.7
hours of waveforms from 43 subjects with 469 seizure activities, to evaluate the detection performance of SeizureTransformer with other traditional and deep-learning algorithms.
The test set of TUSZ is a list of blind EEG signals that are
completely separated from its training set and validation set,
which ensures the generalization of model performance.
We quantify the model’s performance using the area under the receiver operating characteristics(AUROC). For each
continuous EEG recording, the ROC curve plots the true and
false positive rates across all possible decision thresholds,
and the AUC represents the area under the ROC curve,


![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-2-0.png)









![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-2-1.png)



Conv1D & ELU Activation


Pooling


Upsample


Residual Connection


Conv1D & Sigmoid


Res CNN Stack


Position Encoding


Transformer Encoder Stack







Fig. 1. SeizureTransformer Architecture.


TABLE I

M ODEL P ERFORMANCE IN THE SEIZURE DETECTION CHALLENGE 2025.


|Model|Architecture Input Length(s)|Result<br>F1-score Sensitivity Precision FP (per day)|
|---|---|---|
|SeizureTransformer<br>Van Gogh Detector<br>S4Seizure<br>DeepSOZ-HEM<br>HySEIZa<br>Zhu-Transformer<br>SeizUnet<br>Channel-adaptive<br>EventNet<br>Gradient Boost<br>DynSD<br>Random Forest|U-Net & CNN & Transformer<br>60<br>CNN & Transformer<br>N_ ×_ 10<br>S4<br>12<br>LSTM & Transformer<br>600<br>Hyena-Hierarchy & CNN<br>12<br>CNN & Transformer<br>25<br>U-Net & LSTM<br>30<br>CNN<br>15<br>U-Net<br>120<br>Gradient Boosted Trees<br>10<br>LSTM<br>1<br>Random Forest<br>2|0.43<br>0.37<br>0.45<br>1<br>0.36<br>0.39<br>0.42<br>3<br>0.34<br>0.30<br>0.42<br>2<br>0.31<br>0.58<br>0.27<br>14<br>0.26<br>0.6<br>0.22<br>13<br>0.20<br>0.46<br>0.16<br>24<br>0.19<br>0.16<br>0.20<br>4<br>0.14<br>0.06<br>0.20<br>1<br>0.14<br>0.6<br>0.09<br>20<br>0.07<br>0.15<br>0.09<br>6<br>0.06<br>0.55<br>0.04<br>37<br>0.06<br>0.05<br>0.07<br>1|



which summarizes the model’s performance. We compare our
model’s performance using the same evaluation metric under
the TUSZ’s predefined test set with other seizure detection
models, namely, Zhu-Transformer [6], EEGWaveNet [8], and
DCRNN [7], to demonstrate the effectiveness of our proposed
approach. Models used here for the comparison are pre-trained
models based on different training sets. All of these pre-trained
models are implemented by [20] and are publicly available.
As shown in Figure 3, our model demonstrated the highest
performance, with a mean AUROC of 0 _._ 876 and a distribution
tightly concentrated toward higher values.


_D. Application in Seizure Detection Challenge_

The 2025 Seizure Detection Challenge 1, organized as
part of the International Conference on Artificial Intelligence


1 competition website and leaderboard is available in: [https:](https://epilepsybenchmarks.com/challenge/)
[//epilepsybenchmarks.com/challenge/](https://epilepsybenchmarks.com/challenge/)



in Epilepsy and Other Neurological Disorders, provides a
completely blind private dataset consisting of continuous EEG
recordings for evaluation, which makes it an ideal place to
test the performance and generalization of our model fairly.
The test dataset was collected at the EMU of the Filadelfia
Danish Epilepsy Center in Dianalund from January 2018
to December 2020 with the NicoleteOne _[T M]_ v44 amplifier.
The dataset contains 4360 hours of EEG recordings from 65
subjects with various ages, where for each subject, at least
one seizure during the hospital stay with a visually identifiable electrographic correlate to the seizures recorded on the
video. The ground truth labels were annotated by three boardcertified neurophysiologists with expertise in long-term videoEEG monitoring. The F1-score, sensitivity, precision, and false
positive per day were used as the primary ranking criterion to
align with real-world requirements. The event-based scoring
evaluates annotations at the event level by assessing the degree


![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-3-1.png)




















|Model|Total Runtime(s) Runtime(s) per 1-hour EEG|
|---|---|
|SeizureTransformer<br>DCRNN<br>EEGWaveNet<br>Zhu-Transformer|169.96<br>3.98<br>2571.75<br>60.24<br>1690.19<br>39.59<br>3309.51<br>77.53|









![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-3-0.png)

Fig. 2. EEG Signal Processing Pipeline: (a) Brain activity is recorded using
a 19-channel EEG system. (b) A 60-second EEG sample is pre-processed
through normalization, Butterworth bandpass filtering, and 1 Hz & 60 Hz
IIR notch filters to remove noise. (c) After neural network analysis, postprocessing steps—threshold filtering, morphological opening and closing, and
removal of short-duration events—produce the final detection.



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-3-3.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-3-4.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-3-5.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-3-2.png)









Fig. 3. Violin plots illustrating the distribution of AUROC values for
SeizureTransformer, DCRNN, EEGWaveNet, and Zhu-Transformer models
evaluated on the TUSZ v2.0.3 predefined testing set. Mean AUROC scores
for each model are indicated above each plot, with the SeizureTransformer
demonstrating the highest overall performance.


of overlap between predicted and reference events.
As shown in Table I, our model largely outperforms the
other algorithms in terms of F1-score. It is noteworthy that
we set the picking threshold to be 80% in the competition,
which leads to a relative low sensitivity but comes with the
best precision and False Positive rate. Van Gogh Detector
and Zhu-Transformer are window-level classification models
that also take advantage of both convolutional and transformer
encoder units; however, their performance did not reach that
of SeizureTransformer. This points to the beneficial effects of
time-step-level end-to-end learning. Similarly, SeizUnet, like
our model, is a time-step-level classification algorithm using



TABLE II

M ODEL ’ S R UNTIME O VER TUSZ V 2.0.3’ S T ESTING S ET


U-Net; but different to SeizureTransformer, it chooses to add
LSTM layers, instead of transformer encoders, after the U-Net
decoder, instead of embedding into the U-Net, and turns out
to be not as good as our results.


III. D ISCUSSION


_A. Runtime Analysis_


Window-level classification models assign predictions individually to each segmented window. Mapping window labels to the final annotation output followed by the SCORE
compliant [10] that contains the start time and duration time
of a seizure requires the model to segment windows with a
great overlap ratio to ensure the start and stop time’s precision.
This led to tremendous redundant computing and complicated
mapping procedures. On the other hand, the time-step-level
classification models do not require such post-processing steps
as their predictions can directly indicate the onset time and
activity duration. This approach inherently mitigates the redundant computations associated with overlapping windows and
significantly simplifies the annotation pipeline, which makes
this method align more closely with the practical clinical
requirement for efficient automated seizure detection.
We further show our model’s efficiency by comparing the
inference time with other models using TUSZ’s testing set
in Table II. Our model demonstrate the lowest running time
with the ability to handle a one-hour-long recording in 3.98
seconds.


_B. Ablation Study_


The better performance of the proposed method for seizure
detection could be due to several factors. Here, we show
each model component’s necessity by testing multiple partial
models after removing certain components. As shown in
Figure 4, vanilla U-Net has an underwhelming performance
with a low AUROC mean. Solely adding a ResCNN stack or
a transformer stack will marginally improve the model performance but also lead to a bigger variance with some extreme
false cases. By contrast, integrating both the ResCNN and
Transformer stacks produces not only higher mean AUROC
but also reduced variance, indicating that these components
complement each other effectively. These results underscore
the importance of each proposed element in achieving robust
and accurate seizure detection.


_C. Challenge Results_


The competition leaderboard shows a relatively low F1score across every algorithm compared to the results shown in
previously published reviews [21], [22] and the self-reported


![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-4-1.png)





![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-4-3.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-4-4.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-4-5.png)



![](/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/literature/markdown/SEIZURE_TRANSFORMER/SEIZURE_TRANSFORMER.pdf-4-0.png)




|Col1|Col2|
|---|---|
||T|



Fig. 4. Ablation study for SeizureTransformer by drawing AUROC distributions of models that contains partial components. **N** represents a vanilla deep
U-Net without ResCNN and Transformer encoder stack; **R** represents the UNet with ResCNN stack; **T** represents the U-Net with Transformer Stack; **P**
means adding positional encoding before feeding into the transformer stack.


TABLE III

M ODEL PERFORMANCE IN TUSZ’ S PREDEFINED TESTING SET .



tion [12], [25], denoising heart sound signals [26], and Seizure
detection [27].
There are some works exploring combining U-Net with
Transformer together for other fields. For example, in a medical image segmentation task, [28] used self and cross-attention
with U-Net; [29] incorporated hierarchical Swin Transformer
into U-Net to extract both coarse and fine-grained feature
representations. In seismic analysis, [30] proposed a deep
neural network that can be regarded as a U-Net with global
and self-attention but without a residual connection. However,
in the signal processing area, to the best of our knowledge,
there is no existing work to scale U-Net using transformer
blocks. The closest work to this paper [27], where multiple
attention-gated U-Net are used and a following LSTM network
is implemented to fusion results.


_B. Preliminary_


For a continuous EEG waveform, before segmenting it to
uniform windows as training examples, we resample all data to
a common, i.e., 256, sampling rate using the Fourier method

[31], to fix the time resolution for the convolutions in the
model to be meaningful across subjects, and implement a
Gaussian normalization to each channel, calculated by


_x_ _[∗]_ _i_ [= (] _[x]_ _[∗]_ _i_ _[−]_ _[x]_ [¯][)] _[/s]_ _[x]_ _[,]_







_x_ ¯ = [1]

_K_



_K_
� _x_ _i_ _,_


_i_ =1



1
_s_ _x_ = _K −_ 1



_K_


¯

�( _x_ _i_ _−_ _x_ ) [2] _._


_i_ =1


|Scale Model|F-1 Sensitivity Precision|
|---|---|
|Sample-based<br>Gotman<br>EEGWaveNet<br>DCRNN<br>Zhu-Transformer<br>EventNet<br>SeizureTransformer|0.0679<br>0.0558<br>0.0868<br>0.1088<br>0.1051<br>0.1128<br>0.1917<br>0.4777<br>0.1199<br>0.4256<br>0.5406<br>0.3510<br>0.4830<br>**0.5514**<br>0.4286<br>**0.5803**<br>0.4710<br>**0.7556**|
|Event-based<br>Gotman<br>EEGWaveNet<br>DCRNN<br>Zhu-Transformer<br>EventNet<br>SeizureTransformer|0.2089<br>0.6199<br>0.1256<br>0.2603<br>0.4427<br>0.1844<br>0.3262<br>0.5723<br>0.2281<br>0.5387<br>0.6116<br>0.5259<br>0.5655<br>0.6116<br>0.5259<br>**0.6752**<br>**0.7110**<br>**0.6427**|



performance of computing algorithms. To comprehensively
understand the model’s performance, we test our model with
several published algorithms, namely, EventNet [23], ZhuTransformer [6], DCRNN [7], EEGWaveNet [8], and the
Gotman algorithm [24], in the TUSZ’s predefined testing
set using the same evaluation metrics(F1-score, sensitivity,
and precision) implemented by the challenge organizers [20].
The testing tools provide both sample-level and event-level
evaluation. As shown in Table III, while our model keeps
the state-of-the-art performance, all model achieved better F1scores. Such result difference might be due to the distribution
shift between datasets. As described by the organizer, the
private evaluation dataset include recordings from various
ages, and the data was collected by portable EEG amplifiers,
allowing patients to move freely within the building, which
will likely lead to unique attributes in the recording that depart
from the training set.


IV. M ETHODS


_A. Related Work_


U-Net [17] architecture was first proposed in the field of
CV for image segmentation tasks. Considering the temporal
continuity of time series data, such networks have been widely
deployed in various scientific signal processing applications,
such as seismic phase detection [15], sleep-staging classifica


The generated dataset, after slicing, is denoted as _D_ =
( _X_ _, Y_ ) = _{_ ( _x_ _i_ _, y_ _i_ ) _| i_ = 1 _, . . ., N_ _}_, where _N_ represents the
number of training samples. Each input window _x_ _i_ _∈_ R _[T][ ×][d]_

represents a multivariate time series with _T_ = 256 _×_ 60 =
15 _,_ 360 time steps and _d_ = 19 channels. The corresponding
time-step-level label _y_ _i_ _∈{_ 0 _,_ 1 _}_ _[T]_ is a binary, box-shaped
ground truth signal indicating the presence of seizure activity
at each time step.


_C. Network Design_


**Encoder.** We use one-dimensional convolutions along the
time axis to extract local temporal patterns, outputting a
tokenized representation of the signal. Specifically, we use a
convolution-pooling block with various kernel sizes from 11 to
3 to detect features at different temporal scales, capturing both
slow and fast dynamics. This reduces the time step size from
15360 to _T_ _d_ = 512 while increases the channel size from 19
to _k_ _d_ = 480 to compensate the loss of resolution in the time
domain. The ELU function is set as the activation function

after each convolution layer.
**Scaling Embedding.** Followed by [30], after getting the
encoded output, we implement a ResCNN stack first to refine
these tokenized features to yield a better generalization with
better temporal invariance.
We then implement a transformer encoder stack [32] to scale
the model and to capture long-range dependencies across the


tokenized signal. Specifically, the sine and cosine functions of
different frequuencies are used to be positional encodings,


_PE_ ( _pos,_ 2 _i_ ) = _sin_ ( _pos/_ 10000 [2] _[i/T]_ _[d]_ ) _,_


_PE_ ( _pos,_ 2 _i_ +1) = _cos_ ( _pos/_ 10000 [2] _[i/T]_ _[d]_ ) _,_


which can then be summed with the input embedding. The
refined representation, denoted as _Z_, will then be projected
into equally-shaped query, key, and value spaces,


_Q_ = _ZW_ _[Q]_ _, K_ = _ZW_ _[K]_ _, V_ = _ZW_ _[V]_ _,_


and processed with the use of the global-attention mechanism,

_A_ = _softmax_ ( _[Q][K]_ _[T]_ ) _V._

~~_√_~~ _d_ _k_


The attention output is combined with tokens with a residual
connection and layer normalization and a subsequent feedforward network to transform the output with another residual
addition.

Such hierarchical processing scales the model and integrates
both local features and global context, enabling the model to
learn complex temporal dependencies.
**Decoder.** Similar to the encoder, we use a convolutional
decoder to decrypt the compressed information from the center
latent space into a sequence of probability distribution. However, instead of the convolution-pooling block, we upsample
the input with a scale factor of 2 and then with a convolution
to decrease the number of channels and to increase the number

of time steps back to the original window size. Like U-Net,
the residual connections are used between the encoder and

decoder to facilitate efficient gradient flow.
**Training.** The model is trained to produce predictions ˆ _y_ _i_ that
minimize the following objective:


_y_ ˆ _i_ = _f_ _θ_ ( _x_ _i_ ) _,_ _θ ∈_ arg min _L_


Here, we use the Binary Cross-Entropy loss as our training
objective _L_, which measures the dissimilarity between the
predicted and true labels:


ˆ
_L_ (ˆ _y_ _i,j_ _, y_ _i,j_ ) = _−y_ _i,j_ log(ˆ _y_ _i,j_ ) _−_ (1 _−_ _y_ _i,j_ ) log(1 _−_ _y_ _i,j_ )


where _y_ _i,j_ and ˆ _y_ _i,j_ are the ground truth and predicted labels,
respectively, for sample _i_ at time step _j_ .


V. L IMITATION


While there has been a rich literature of research on epileptic
seizure detection and prediction, there is more work to be
done to generalize the algorithms to anatomically different
types of epilepsy, different ambulatory settings for recordings.
This is evident from the gaps between the training-validations
v.s. testing F1-scores of the work presented in this paper. Our
demonstrates a high F1-score over other data sets. However,
its F1-score is lower on the withheld test data set while it still

out performs the competing ones with a significant difference.
Thus, future work will focus on understanding the differences
in the data distributions between training and test data sets to
improve our model.



VI. D ATA AVAILABILITY


We used the following publicly available datasets in this
work for training our model. The test set used in the competition was not made publicly available at the time of this
write-up.


_•_ _Siena Scalp EEG Database_ : The database consists of
EEG recordings of 14 patients acquired at the Unit
of Neurology and Neurophysiology of the University
of Siena. Subjects include 9 males (ages 25-71) and
5 females (ages 20-58). Subjects were monitored with
a Video-EEG with a sampling rate of 512 Hz, with
electrodes arranged on the basis of the international 10-20
System. Most of the recordings also contain 1 or 2 EKG
signals. The diagnosis of epilepsy and the classification
of seizures according to the criteria of the International
League Against Epilepsy were performed by an expert
clinician after a careful review of the clinical and elec
[trophysiological data of each patient. License: https://](https://physionet.org/content/siena-scalp-eeg/view-license/1.0.0/)
[physionet.org/content/siena-scalp-eeg/view-license/1.0.0/](https://physionet.org/content/siena-scalp-eeg/view-license/1.0.0/)

_•_ _TUH EEG Seizure Corpus v2.0.3_ : This database is a
subset of the TUH EEG Corpus that was collected from
archival records of clinical EEGs at Temple University
Hospital recorded between 2002 – 2017. From this large
dataset, a subset of files with a high likelihood of containing seizures was retained based on clinical notes and
on the output of seizure detection algorithms. V2.0.0
contains 7377 .edf files from 675 subjects for a total
duration of 1476 hours of data. The files are mostly
short (avg. 10 minutes). The dataset has a heterogeneous
sampling frequency and number of channels. All files are
acquired at a minimum of 250 Hz. A minimum of 17 EEG
channels is available in all recordings. They are positioned according to the 10-20 system. The annotations are
provided as .csv and contain the start time, stop, channel,
[and seizure type. License: https://isip.piconepress.com/](https://isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf)
[projects/nedc/forms/tuh eeg.pdf.](https://isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf)


VII. C ODE AVAILABILITY


[Our source code and model are available at https://github.](https://github.com/keruiwu/SeizureTransformer)
[com/keruiwu/SeizureTransformer.](https://github.com/keruiwu/SeizureTransformer)


R EFERENCES


[1] L. Hirsch, E. Donner, E. So, M. Jacobs, L. Nashef, J. Noebels, and
J. Buchhalter, “Abbreviated report of the nih/ninds workshop on sudden
unexpected death in epilepsy,” _Neurology_, vol. 76, no. 22, pp. 1932–
1938, 2011.

[2] A. Van de Vel, K. Cuppens, B. Bonroy, M. Milosevic, K. Jansen,
S. Van Huffel, B. Vanrumste, P. Cras, L. Lagae, and B. Ceulemans,
“Non-eeg seizure detection systems and potential sudep prevention: state
of the art: review and update,” _Seizure_, vol. 41, pp. 141–153, 2016.

[3] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet:
Temporal 2d-variation modeling for general time series analysis,” _arXiv_
_preprint arXiv:2210.02186_, 2022.

[4] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,”
_arXiv preprint arXiv:2310.06625_, 2023.

[5] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “Fedformer:
Frequency enhanced decomposed transformer for long-term series forecasting,” in _International conference on machine learning_ . PMLR,
2022, pp. 27 268–27 286.


[6] Y. Zhu and M. D. Wang, “Automated seizure detection using transformer
models on multi-channel eegs,” in _2023 IEEE EMBS International_
_Conference on Biomedical and Health Informatics (BHI)_ . IEEE, 2023,
pp. 1–6.

[7] S. Tang, J. A. Dunnmon, K. Saab, X. Zhang, Q. Huang, F. Dubost,
D. L. Rubin, and C. Lee-Messer, “Self-supervised graph neural networks
for improved electroencephalographic seizure analysis,” _arXiv preprint_
_arXiv:2104.08336_, 2021.

[8] P. Thuwajit, P. Rangpong, P. Sawangjai, P. Autthasan, R. Chaisaen,
N. Banluesombatkul, P. Boonchit, N. Tatsaringkansakul, T. Sudhawiyangkul, and T. Wilaiprasitporn, “Eegwavenet: Multiscale cnnbased spatiotemporal feature extraction for eeg seizure detection,” _IEEE_
_Transactions on Industrial Informatics_, vol. 18, no. 8, pp. 5547–5557,
2021.

[9] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with
selective state spaces,” _arXiv preprint arXiv:2312.00752_, 2023.

[10] S. Beniczky, H. Aurlien, J. C. Brøgger, L. J. Hirsch, D. L. Schomer,
E. Trinka, R. M. Pressler, R. Wennberg, G. H. Visser, M. Eisermann
_et al._, “Standardized computer-based organized reporting of eeg: Score–
second version,” _Clinical Neurophysiology_, vol. 128, no. 11, pp. 2334–
2346, 2017.

[11] A. Bagnall, H. A. Dau, J. Lines, M. Flynn, J. Large, A. Bostrom,
P. Southam, and E. Keogh, “The uea multivariate time series classification archive, 2018,” _arXiv preprint arXiv:1811.00075_, 2018.

[12] H. Li and Y. Guan, “Deepsleep convolutional neural network allows
accurate and fast detection of sleep arousal,” _Communications biology_,
vol. 4, no. 1, p. 18, 2021.

[13] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning
with a unified text-to-text transformer,” _Journal of machine learning_
_research_, vol. 21, no. 140, pp. 1–67, 2020.

[14] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy,
V. Stoyanov, and L. Zettlemoyer, “Bart: Denoising sequence-to-sequence
pre-training for natural language generation, translation, and comprehension,” _arXiv preprint arXiv:1910.13461_, 2019.

[15] W. Zhu and G. C. Beroza, “Phasenet: a deep-neural-network-based seismic arrival-time picking method,” _Geophysical Journal International_,
vol. 216, no. 1, pp. 261–273, 2019.

[16] C. Chatzichristos, J. Dan, A. M. Narayanan, N. Seeuws, K. Vandecasteele, M. De Vos, A. Bertrand, and S. Van Huffel, “Epileptic seizure
detection in eeg via fusion of multi-view attention-gated u-net deep
neural networks,” in _2020 IEEE Signal Processing in Medicine and_
_Biology Symposium (SPMB)_ . IEEE, 2020, pp. 1–7.

[17] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks
for biomedical image segmentation,” in _Medical image computing and_
_computer-assisted intervention–MICCAI 2015: 18th international con-_
_ference, Munich, Germany, October 5-9, 2015, proceedings, part III 18_ .
Springer, 2015, pp. 234–241.

[18] V. Shah, E. Von Weltin, S. Lopez, J. R. McHugh, L. Veloso, M. Golmohammadi, I. Obeid, and J. Picone, “The temple university hospital
seizure detection corpus,” _Frontiers in neuroinformatics_, vol. 12, p. 83,
2018.

[19] P. Detti, “Siena scalp eeg database,” _PhysioNet. doi_, vol. 10, p. 493,
2020.

[20] J. Dan, U. Pale, A. Amirshahi, W. Cappelletti, T. M. Ingolfsson,
X. Wang, A. Cossettini, A. Bernini, L. Benini, S. Beniczky _et al._,
“Szcore: Seizure community open-source research evaluation framework
for the validation of electroencephalography-based automated seizure
detection algorithms,” _Epilepsia_, 2024.

[21] S. Supriya, S. Siuly, H. Wang, and Y. Zhang, “Epilepsy detection from
eeg using complex network techniques: A review,” _IEEE Reviews in_
_Biomedical Engineering_, vol. 16, pp. 292–306, 2021.

[22] M. K. Siddiqui, R. Morales-Menendez, X. Huang, and N. Hussain, “A
review of epileptic seizure detection using machine learning classifiers,”
_Brain informatics_, vol. 7, no. 1, p. 5, 2020.

[23] N. Seeuws, M. De Vos, and A. Bertrand, “Avoiding post-processing
with event-based detection in biomedical signals,” _IEEE Transactions_
_on Biomedical Engineering_, 2024.

[24] J. Gotman, “Automatic recognition of epileptic seizures in the eeg,”
_Electroencephalography and clinical Neurophysiology_, vol. 54, no. 5,
pp. 530–540, 1982.

[25] M. Perslev, M. Jensen, S. Darkner, P. J. Jennum, and C. Igel, “U-time: A
fully convolutional network for time series segmentation applied to sleep



staging,” _Advances in neural information processing systems_, vol. 32,
2019.

[26] A. Mukherjee, R. Banerjee, and A. Ghose, “A novel u-net architecture
for denoising of real-world noise corrupted phonocardiogram signal,”
_arXiv preprint arXiv:2310.00216_, 2023.

[27] M. R. Islam, X. Zhao, Y. Miao, H. Sugano, and T. Tanaka, “Epileptic
seizure focus detection from interictal electroencephalogram: a survey,”
_Cognitive Neurodynamics_, vol. 17, no. 1, pp. 1–23, Feb 2023.

[28] O. Petit, N. Thome, C. Rambour, L. Themyr, T. Collins, and L. Soler,
“U-net transformer: Self and cross attention for medical image segmentation,” in _Machine Learning in Medical Imaging: 12th International_
_Workshop, MLMI 2021, Held in Conjunction with MICCAI 2021,_
_Strasbourg, France, September 27, 2021, Proceedings 12_ . Springer,
2021, pp. 267–276.

[29] A. Lin, B. Chen, J. Xu, Z. Zhang, G. Lu, and D. Zhang, “Ds-transunet:
Dual swin transformer u-net for medical image segmentation,” _IEEE_
_Transactions on Instrumentation and Measurement_, vol. 71, pp. 1–15,
2022.

[30] S. M. Mousavi, W. L. Ellsworth, W. Zhu, L. Y. Chuang, and G. C.
Beroza, “Earthquake transformer—an attentive deep-learning model for
simultaneous earthquake detection and phase picking,” _Nature commu-_
_nications_, vol. 11, no. 1, p. 3952, 2020.

[31] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy,
D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright _et al._,
“Scipy 1.0: fundamental algorithms for scientific computing in python,”
_Nature methods_, vol. 17, no. 3, pp. 261–272, 2020.

[32] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” _Advances in_
_neural information processing systems_, vol. 30, 2017.


