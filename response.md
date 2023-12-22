That's quite a diversity of topics you're studying! Here's a brief overview:

1) Finite State Text Processing, Morphology, Pynini:
Finite state machine is a model of computation with highly structured set of transitions between states. Used in text processing, finite state machines can parse, filter, and transform text data. Morphology, the study of words and how they're formed, often uses finite state machines. Pynini, conversely, is a Python library for creating finite state transducers.

2) Text Normalization, Spelling:
Text normalization involves transforming text into a consistent form - for example, lowercasing, stemming, or removing punctuation. Pynini can be used for this task. Spelling correction is another aspect of text normalization; it involves identifying and correcting spelling mistakes.

3) FSA, Reg Expressions, Conditional Prob., Bayes:
FSA (Finite State Automata) and regular expressions are both tools for pattern matching. Conditional probability is a measure of the probability of an event given that another event has occurred. Bayes' theorem is a way to calculate conditional probabilities.

4) Text Classification - Traditional Methods and Neural Methods:
Traditional methods of text classification include techniques like Naive Bayes and Logistic Regression, which often use features derived from the word counts in the text. Neural methods for text classification, such as Multilayer Perceptrons (MLP) and Convolutional Neural Networks (CNN), can learn complex patterns and are more flexible.

5) Sequence Labeling:
This involves predicting a label for each item in a sequence. Models like Hidden Markov Models, Recurrent Neural Networks (RNNs), and Long Short Term Memory (LSTM) Networks are common for this task. 

6) Sequence-to-Sequence Models:
These models involve mapping an input sequence to an output sequence. Encoder-Decoder models are common for tasks like machine translation, where an encoder reads the input sequence and a decoder generates the output. Attention mechanisms help the model focus on different parts of the input sequence.

7) Transformers:
A transformer is a type of model that uses self-attention mechanisms to capture dependencies between all words in the input sequence, regardless of their distance apart.

8) Pretrained Language Models:
These models are trained on a large text corpus and can be fine-tuned for a variety of tasks. They capture a lot of language knowledge and can often greatly improve performance.

9) Syntax, Context-Free Grammars and Parsing:
Syntax refers to the structure of sentences. Context-Free Grammars are a tool for describing the possible structures a sentence can have. Parsing is the process of taking a sentence and producing its structure according to a given grammar.

10) Semantic Role Labeling, Lexical Semantics:
Semantic role labeling involves identifying the semantic roles of different parts of the sentence. Lexical semantics looks at the meaning of words and their relations.

11) Topic Modeling:
This involves uncovering the underlying topics in a set documents. Methods like Latent Dirichlet Allocation are common.

12) Word Embeddings:
Word embeddings are a way of representing words as dense vectors, capturing meaning based on the word's usage.

13) Discourse and Coreference:
Discourse deals with how language is used in conversation. Coreference resolution is about determining when two phrases in a text refer to the same entity.

14) Summarization:
Summarization involves taking a long document and reducing it to a shorter summary while retaining the main points.

Each of these topics can be applied in various scenarios depending on your use case. For example, FSA can be used for text search, spelling correction for text cleaning, Text classification for sentiment analysis or spam detection, Sequence labeling for part of speech tagging or named entity recognition, Syntax parsing for generating parse trees and understanding sentence structure etc.

### Quiz #1:
Sure, here are three questions to help you better understand Chunking, Dependency Parsing, and Treebanks:

1) What is chunking in Natural Language Processing, and provide an example of its application.

Answer: Chunking in NLP is a process of extracting phrases from unstructured text by grouping words into meaningful chunks. Instead of identifying individual words, chunking groups them based on their part of speech tags. For example, in the sentence "The quick brown fox jumps over the lazy dog”, a chunker could group "The quick brown fox" as a noun phrase and "over the lazy dog" as a prepositional phrase. A common application of chunking can be seen in information extraction tasks where it's used to identify named entities like persons, organizations, location, etc.

2) Explain dependency parsing and its significance in understanding sentence structure.

Answer: Dependency parsing is an approach to analyzing the grammatical structure of a sentence based on the dependencies between the words. Each word in the sentence has a dependency relation to another word, typically expressing relations like subject, object, or modifying relation. For example, in the sentence "Amy tossed the ball", "Amy" is the subject of the verb "tossed", and "the ball" is the object of the verb. This structure provides a more direct mapping between syntax and semantics of the sentence, leading to a better understanding of the sentence's meaning.

3) What are Treebanks and how are they used in syntactic analysis of text?

Answer: Treebanks are a type of linguistic database where sentences are annotated with syntax (and sometimes semantic) structure. Sometimes the syntactic structure is a tree, hence the name treebank. They are commonly used in the development and evaluation of parsing algorithms. For example, a statistical parser may be trained based on the patterns observed in a treebank. Additionally, treebanks can be used to compute statistics about language use, study linguistic phenomena, and as input for other NLP algorithms like machine translation systems. An example of a treebank is the Penn Treebank, which annotates the Wall Street Journal Corpus.


### Question - Summarise this in bullet points: https://web.stanford.edu/~jurafsky/slp3/17.pdf:
This document provides comprehensive material on the topic of Information Extraction (IE), Relation Extraction, and Sentiment Analysis. Here are the key takeaways in bullet points:

Information Extraction:
- Information Extraction is the process of automatically extracting structured information from unstructured data, like text.
- It involves various tasks such as Named Entity Recognition, Coreference Resolution, and Relation Extraction.
- Named Entity Recognition (NER) is the task of identifying proper names in text.
- Coreference resolution is as the task of determining when two expressions in a text refer to the same entity.

Relation Extraction:
- Relation Extraction is the task of identifying and classifying semantic relationships between entities in text.
- There are various approaches to relation extraction, including Rule-based, Supervised, Semi-supervised, and Unsupervised.
- The main challenge of relation extraction is dealing with the vast variety of ways in which a particular relation can be expressed.

Sentiment Analysis:
- Sentiment Analysis is the field of study that analyzes people's opinions, sentiments, evaluations, appraisals, and emotions towards entities such as products, services, organizations, individuals, issues, events, and their attributes.
- It includes tasks like identifying sentiment polarity, subjectivity, intensity, aspect-based sentiment analysis, and emotion detection.
- It is widely applied in areas like business intelligence, social media monitoring, product analysis, and market research.

Topic Detection:
- Topic Detection is the task of automatically identifying the main themes that occur in a text.
- It enables the analysis of large volumes of data, for example in document clustering, document summarization, or improving information retrieval.
- Latent Dirichlet Allocation (LDA) is one of the most popular methods for Topic Detection.

Event Extraction:
- Event Extraction involves identifying instances of a specified type of event in text and determining the roles of entities in the event.
- Events are generally represented as a structured frame or template, specifying the type of event and the roles of entities involved.

### Test #1:
Here are some questions based on the material:

1) A company receives a tweet saying "I love your product #Blessed". What type of Information Extraction method would help identify the sentiment expressed in this tweet and what does it express? 

Answer: Sentiment Analysis would help identify the sentiment. This tweet expresses a positive sentiment.

2) What type of NLP task would you use to correctly recognize "Apple" as a tech company rather than a fruit in the sentence "Apple has just launched a new product"?

Answer: Named Entity Recognition would be used to identify "Apple" as a tech company.

3) In the sentence, "Julia passed the ball to Sarah. She scored a goal", what is the coreference in this sentence?
 
Answer: The coreference in this sentence is "She" referring to "Sarah".

4) If you are given a large set of customer reviews about a product and you want to find out what specific features are being discussed, what NLP task would you apply?

Answer: This can be achieved by Aspect-based Sentiment Analysis or Topic Detection.

5) If you had to pick between a Count-Based Trigram Language Model and an RNN-based Language Model, which would be more capable of handling long-range dependencies in sentences, and why?

Answer: An RNN-based Language Model would be more capable of handling long-range dependencies because, unlike the Trigram Model that only considers the last two words, an RNN can consider all previous words in the sentence.

6) For the sentence "Bobby gave the book to Julie", what is the relation and the entities involved if we were doing relation extraction?

Answer: The relation is 'gave' and the entities involved are 'Bobby' and 'Julie'.

7) What is Treebank and how is it helpful in syntactic analysis of text?

Answer: A Treebank is a linguistic database where sentences are annotated with syntax and sometimes semantic structure. It is helpful in syntactic analysis as it helps in the development and evaluation of parsing algorithms.

8) What type of language model - Count-Based Trigram, Neural Trigram, or RNN-Based model - would most likely give the sentence "The cat, that was very old and very wise, sat on the mat" the highest probability and why?

Answer: The RNN-Based model would be most likely to give this sentence the highest probability because it is capable of handling the long-range dependencies present in this sentence.

9) In the sentence, "Paul gave the ball to Peter", if we run an event extraction, how would it present the resulting structure?

Answer: The event is 'gave', with 'Paul' as the giver, 'the ball' as the object being given, and 'Peter' as the receiver.

10) Given the set of sentences "I love the new Xbox. The graphics are mind-blowing and the controller feels great in the hands!", assuming you're using sentiment analysis, which aspects are being mentioned and what sentiment is being expressed about them?

Answer: The aspects being mentioned are the 'graphics' and the 'controller' of the new Xbox. The sentiment is positive for both aspects.
