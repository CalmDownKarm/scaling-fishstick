# Task 1 Activity Prediction
For the activity prediction problem, the activity variable is a real-valued label that typically ranges from -5 to +8. As a result, I approached the issue as a standard regression problem. The data is slightly skewed - with a mean of 0 and standard deviation of 1, about 2000 samples out of the 100000 provided have an activity greater than 3 standard deviations.

Regarding models, there are several possible approaches to consider. My primary approach is an RNN with attention. This model has a few advantages which is why I favour this solution.
1. It's relatively small, under 10 million parameters.
1. It's a single architecture that can be used for both tasks, thus allowing for some amount of code reuse
1. For this scale of data, experimentally, RNNs seem to perform equally well as small transformer models, but are significantly simpler to train. 


## RNN Model - Discussion Questions
### Describe your model
To prepare the sequences for model input, I tokenize each amino acid into a separate token and append indicator tokens to signify the start and end of each sequence. These tokens then feed into an embedding layer, which leads to a stack of LSTMs. The output from the LSTM is then passed to a multi-head attention layer, with the output from the attention layer ultimately feeding into a single or stack of linear layers.

For the regression task, I first pretrained the RNN with a language modelling objective, in this case, the linear layer is a simple linear decoder. For this, I concatenate all the sequences together, randomly sample contiguous protein sequences and try to predict the subsequent sequence. Thus the decoder outputs a tensor of shape (batchsize, sequence length, vocab size) and trained with a cross entropy loss. Once this model has been adequately trained, I replace the linear decoder with a stack of linear layers that output a tensor of shape (batchsize, 1). This is then trained with a MSE loss since RMSE is not differential in all places.

There's a few training tricks in order to make the process smoother - firstly, I gradually unfroze the encoder layers in the regression model, I use FastAI's LR finder approach to set the learning rate and finally use gradient clipping to ensure gradients don't go over (0.5-0.7) per batch. Experimentally, gradient clipping didn't have a very strong effect, so can potentially be removed. 

### Performance
I selected the following metrics-
* Root Mean Squared Error (RMSE) - This metric is slightly more interpretable than the mean squared error since the error is in the same range as the target variable. For example, an RMSE of 0.1 means the model has an average error of +/-10% on the activity variable. This metric is beneficial since a mean squared error between 0 and 1 can be lower than the RMSE. 

* R2 Score or Coefficient of Determination - This metric reflects the variance of the activity variable that has been explained by the independent variables in the model. However, a downside of this metric is that it assumes all features are independent, which may not be valid for protein sequences. This could lead to an overinflation of the R2 score and overestimate the effectiveness of our model. Nonetheless, it is still a good baseline metric to use.

* Mean Absolute Error - this metric is also on the same scale as the target variable with the advantage of being less vulnerable to outliers. The training data has 2% of samples lying outside 2 standard deviations of the mean. Thus this metric is not essential and is presented mostly as a "good to know"

Based on a randomly sampled hold-out validation set of 20k samples, my model achieved an **RMSE of 0.45, R2 Score of 0.79, MAE of 0.33**.

### Generalization
On the validation set, the model predictions are strongly correlated with the target variable (Pearson Coefficient 0.89). Correlation between the validation set and the predictions aren't completely indicative of generalization performance, simply because we don't know if the test set comes from the same distribution as the validation/training set, however if we make the assumption that the test set is similar to the training set, I would expect the model to generalize relatively well. A key issue is the model's performance, An RMSE of ~0.5 means that the model's predictions have an error equivalent to half the standard deviation for this data. This implies that even if the model can generalize, its predictions will still be noisy.


# Task2 Sequence Prediction
The Sequence Prediction task can be viewed as a conditional text generation problem where the generated sequences are conditioned on the activity variable. The performance of this task can be measured primarily using Token Accuracy. While metrics such as ROUGE or BLEU are commonly used in language generation tasks, I believe the order of amino acids plays a significant role in protein sequences. As a result, Token Accuracy is the most appropriate metric to use for this task.

## RNN Model - Discussion Questions

### Easier or harder
The Sequence Prediction task is considerably more challenging for several reasons. Firstly, there is the generation task itself, which can be modeled as a multi-label classification problem or a sequence generation problem. The sequence generation problem is a better approach since the sequence of amino acids in the generated proteins is important. As a result, this is a causal language modeling task.
Secondly, it is difficult to map a real-valued number to a sequence of proteins due to the limited information provided by a single real value for conditioning the model. This lack of information presents a significant challenge for generating accurate sequences.

### My Method
For the Sequence Prediction task, I used an RNN model with a similar structure as the first task. However, there were a few differences in this model:
I copied over the embeddings from the first model to this model, which helped to improve its generalization ability.
Additionally, the activity variable was fed to the model by concatenating it with the input embeddings along the embedding dimension and passing it to the LSTM layer. 

During training, I choose a random position in the protein sequence, select a subsequence of tokens from that position and ask it to predict the last token in this sequence. This approach ensures that the model saw missing contexts of every token in the samples given enough training epochs. 
After training, I used simple top-k sampling to generate new sequences of positions, however I realized that the model is generating a lot of repetitions. I tried adding a small epsilon value(1e-4) to the activity variable for each successive token to decrease the chance of degenerating to repetitive sequences. [preds/generated_sequences_rnn.csv](preds/generated_sequences_rnn.csv) contains examples of generated sequences and activities sampled from the validation set.



### Performance
Training this model was challenging, and the loss saturated quickly, regardless of whether I increased or decreased the model's size. On a validation set of 20,000 randomly sampled samples, it achieved a **Strict Accuracy of 4% and a Loose Token Accuracy of 47%**.
I calculated token accuracy in two ways:
* Strict Token Accuracy: I generated a sequence for each activity and compared it to the target sequence. This approach is similar to a zero-one-loss normalized over 12 tokens. I then averaged this accuracy metric over the entire set of predictions to obtain a score.

* Loose Token Accuracy: I rounded up activities to 3 decimal places and then grouped sequences by the rounded-off activity to create a set of potential sequences for a single activity. I then compared each token in my predicted sequence against any tokens at the same position in the target sequences. This metric is closer to a Hamming score for multi-class classification.
Qualitatively, this model's performance is considerably lacking, it tends to predict a large number of duplicate tokens.



### A Note on my Approach
RNNs are definitely a little dated as neural net approaches come - and in particular 
With the advent of large language models, they seem well suited to these sorts of problems, however thanks to limited compute/budgets for externally hosted models, I've largely avoided looking at these solutions. For task 1 specifically, I don't believe these models are well suited for regression in the text completion form. It's possible to use FlanT5 or some other foundation model as a feature extractor for the protein sequence, but I thought it was better to focus on the RNN because of much faster iteration speed.

Another option would be to leverage a pretrained BERT style model such as [prot_bert](https://huggingface.co/Rostlab/prot_bert), such as those trained on external protein sequence data. Although these models should perform exceptionally well and can be run on free GPUs, they are pre-trained on significantly longer protein sequences. They may need a lot of data for sequences of only 12 tokens. Additionally, fine-tuning two different models, a BERT-style encoder model for the first task and a GPT2-style decoder model for the second, would again run into computing limitations.

#### ChatGPT
For curiosity I evaluated ChatGPT on Task 2. After rounding the activities to 3 decimal places, I selected a subset of 1000 sequence, activity pairs from the validation set. For each activity datapoint, I sampled 20 training examples, and passed them to ChatGPT as exemplars. The model does perhaps unsurprisingly well. In places where the model predicts sequences longer than 12 amino acids, I trunctate the predictions to the first 12. With this, it acheives **Strict Accuracy of 5% and a Loose Token Accuracy of 37%** (Note - this is a thousand samples compared to the RNN which is over twenty thousand samples.). Examples of Chatgpt produced sequences are in [preds/generated_sequences_chatgpt.csv](preds/generated_sequences_chatgpt.csv), qualitatively, the model performs worse than my RNN because it occasionally generates tokens which are not valid like  `'B', 'J', 'O', 'U', 'X', '[', ']'`




### Code Organization and Reproducability
* all models are defined in [model.py](model.py)
* [regression.py](regression.py) contains the code to train and make a prediction for the regression model. 
* [sequence_prediction.py](sequence_prediction.py) contains the code to train and make predictions for the sequence prediction task
* [chatgpt_inference.py](chatgpt_inference.py) contains code to run inference for ChatGPT.
There's some amount of code duplication as I've tried to isolate these scripts to be able to run independently, however major pipeline components are shared either in `model.py` or `utils.py`
* The model checkpoints are about 60mb each, so I uploaded them to google drive and they can be downloaded from [this link](https://drive.google.com/drive/folders/1SbksI586ZZ9B73msiDzijZzqYFfxScxt?usp=sharing)
* I exported my conda environment in [environment.yaml](environment.yaml), however a minimal set of dependences is in [dep.yaml](dep.yaml) which should cover all the major requirements without pinned version numbers.