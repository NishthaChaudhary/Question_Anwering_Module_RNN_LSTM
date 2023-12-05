# Question_Anwering_Module

The focus in this project is Question Answering: answering a question based on the number of facts. This project will involve using memories: facts that are already stated are stored in memory, and the question refers to past information to produce an answer. I will demonstrate this using different approaches: “flat memory” approach: RNN and LTSM; and “responsive memory” approach: end-to-end memory. 
These approaches will teach a machine to associate answers to questions irrespective of having several irrelevant facts/stories in between the question and the relevant fact.  

# Abstract:
As my project involves textual data (Facts & Questions) that will have different lengths each time; and use memory of the past facts. I will implement deep learning models.
Language: Python 3.6
Deep Learning Library : Keras (provide high- level building blocks for developing deep-learning models)
Backend Implementation: TensorFlow

# Scenario:

Let us say I am working on a language understanding module for a chatbot. The chatbot must be able to answer questions about historical 
facts I hand it through a chat window: it must be able to refer to older information in order to answer a question. Specifically, every 
question can be answered by exactly one statement that occurred in the past. I have at hand a large dataset of hand-annotated questions 
linked to supporting facts, and a set of candidate architectures that allow me to reason about memory. Using RNN, LSTM, and end-to-end 
memory networks, how can I implement this chatbot module? Before that, it is time to play with data.

# DATA AND DATA PROCESSING

Facebook bAbI is a set of 20 QA tasks, each consisting of several context-question-answer triplets, prepared and released by Facebook. 
The bAbI dataset is composed of synthetically generated stories about activity in a simulated world. In addition to the story, the context
includes pointers to the relevant supporting facts, the sentences within the story that are necessary for answering the question. This 
allows for strongly supervised learning, where the supporting facts are provided during training, as well as the more common weakly 
supervised learning, where training makes use of the story, question and answer, but does not use the supporting facts.

## Tokenization

Three list of vectors are created:
* A list holding all facts as vectors.
* A list of vectorized questions.
* A list of labels: word indices referring to the word that is answer to a question.

I will represent every list of facts (either including intervening, irrelevant facts or just the fact(s) holding the answer to the 
question) as one big vector. So, I will basically concatenate the entire list of facts to one big string and tokenize that string 
(convert it to a numerical vector). 

## Vectorization

After the relevant facts for answering the question have been determined, I will vectorize the facts, the question, and the answer, 
and append results to designated output variables for the entire training and test data sets (X for facts, Q for questions, and y for
answers). 
* X: Array of vectorized facts
*	Q: Array of vectorized questions
*	y: Array of one-hot-encoder vectors representing the answer word.

# QUESTION ANSWERING WITH SEQUENTIAL MODELS

## RNNs for Question Answering:
A recurrent neural network (RNN) processes sequences by iterating through the sequence elements and maintaining a state containing 
information relative to what it has seen so far. In effect, an RNN is a type of neural network that has an internal loop.
In this project, a branching model with two RNNs are implemented. The two RNNs handle the facts (stories) and the question respectively.
Their output is merged by concatenation and sent through a Dense layer that produces a scalar of the size of our answer vocabulary, 
consisting of probabilities. 

## LSTMs for Question Answering:
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. 
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically
their default behaviour, not something they struggle to learn!
The LSTM model is similar to the RNN model. It has two LSTM layers processing stories and questions, and the results (the output layers)
are merged by concatenation.

 ## END-TO-END MEMORY NETWORKS FOR QUESTION ANSWERING:
 End-to-end memory networks embody responsive memory mechanisms. In the context of this project, rather than just teaching a network 
 to predict an answer word index from a combined vector of facts and a question, these networks produce a memory response of a series 
 of facts (a story) to the question posed, and use that response to weigh the facts vector.
 
* Facts (here called 'sentences') are embedded with an embedding A.
* Question q is embedded with an embedding B. 
*	Simultaneously, facts are embedded with a separate embedding C. 
*	The 'response' of the memory bank consisting of the embedded facts is computed by first deriving an inner product of the embedded 
facts with the embedded question, after which a softmax layer produces probabilities.
*	These trainable probabilities are tuned during training through backpropagation. 
*	Finally, the probabilities are combined with the fact embedding produced by embedding C with a weighted sum operation. 
*	Embedded question is combined through concatenation with the weighted sum. The result of this is sent to a final weights layer feeding
into a dense output layer, encoding the word index of the answer.

# IMPLEMENTATION
SimpleRNN, LSTM and end-to-end memory network model was evaluated for epochs=30 with following context size on 9000 Training samples and
1000 validation samples:

* Context Size=0; 1 irrelevant fact; maximum context length=12 words
*	Context Size=2; 2 irrelevant fact; maximum context length=18 words
*	Context Size=4; 4 irrelevant fact; maximum context length=30 words
*	Context Size=6; 6 irrelevant fact; maximum context length=41 words
*	Context Size=8; 8 irrelevant fact; maximum context length=52 words
*	Context Size=All; All irrelevant fact; maximum context length=58 words

# RESULT
*	Running all the models, I observed a clear effect of context size on the training performances.
*	SimpleRNN faced the issue of the vanishing gradient descent.
*	Memory networks show a less steep degree in performance when context size is increased and reach a better score in the all-facts 
situation.
*	While not perfect, end-to-end memory networks perform much better on our Question Answer module.

# CONCLUSION
In this project, I have gone through three approaches to Question Answering module of a chatbot. It was observed that RNNs perform 
worse than LSTM in remembering long sequences of words and that memory end-to-end networks outperform LSTMs.
