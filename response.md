Given the range of topics, there will be quite a lot to cover here. I'll try to summarize the key points, but this is really a broad overview so you may need to dive into specific areas in more depth. 

1. Finite State Text Processing involves the use of systems that process input strings based on a set of defined states and transitions. It is often applied in Natural Language Processing (NLP) for things like text normalization and spell checking. 

2. Morphology in linguistics is the study of words, their internal structure, and how they are formed. For example, 'jumping' is formed from 'jump' and 'ing'. 

3. Pynini is a Python library that compiles finite-state grammars written in a regular-expression-like syntax. 

4. FSA (Finite State Automata) and Regular Expressions are both used to recognize patterns in strings. These can be used in text normalization or spell-checking.

5. Conditional Probability and Bayes' theorem, are the backbone of probabilistic model verification in NLP, such as in spam filtering or document classification.

6. Text Normalization is the process of transforming text into a canonical (standard) form. For example, converting numbers into words like '9' to 'nine'.

7. Language Models predict the next word in a sequence given the history of previous words. They can be Traditional (based on n-grams, smoothed or back-off models) or Neural (machine learning-based models capable of learning word dependencies).

8. Text Classification is the task of assigning predefined categories to a text. This can be done traditionally with methods like Naive Bayes and Logistic Regression, or more advanced methods like Feedforward Neural Networks, Multi-Layer Perceptron (MLP), and Convolutional Neural Networks (CNN).

9. Sequence Labeling is the task of predicting labels for each member of a sequence. Markov Models and Hidden Markov Models are often used for this. POS tagging and NER (Named Entity Recognition) are some applications.

10. Sequence-to-sequence models are deep learning models where both input and output are sequences. "Encoder-Decoder" and "Attention" models are examples of such models used for tasks like Machine Translation.

11. Transformers are a specific type of model used in NLP notable for their self-attention mechanism which lets them consider different words in a sentence to establish context. 

12. Pretrained language models like BERT, GPT, or ELMo are general-purpose models trained over a large amount of unlabelled text data (like the entire internet). 

13. Syntax, Context-Free Grammars, Parsing, Chunking, Treebanks, Sequence Modeling are all concerned with understanding the structure of language.

14. Semantics is the study of meaning. This can be broken down into lexical semantics (the meaning of individual words) and compositional semantics (how meanings combine).

15. Semantic Role Labeling is a task in NLP which involves detecting the semantic relationships between entities in a sentence.

16. Topic Modeling is a type of statistical model for discovering abstract topics that occur in a collection of documents.

17. Word embeddings translate words into vectors of real numbers so that the word's meaning can be understood by the machine. 

18. Discourse and Coreference are about understanding the larger context of a conversation or a paragraph, and how different parts refer to each other.

19. Summarization is the task of making a shorter version of a document while keeping the main points. 

These concepts are all integral parts of NLP and computational linguistics, and they interact in many ways. They're used in various applications like machine translation, question answering systems, document or sentiment classification, information extraction and so forth.

### Test #1:
Sure, here are the questions:

1. A sentence reads "The dog chase the cat". Correct the sentence using Finite State Automata approach and explain how you did it.

2. Consider the word "unhappiness". Break it down into its morpheme components and explain the role of each morpheme.

3. You are working with the Pynini library and are tasked to build a finite state grammar to recognize email addresses. Explain your approach on doing this.

4. The probability of receiving a spam email is 30% and the probability of a spam email containing the word "prize" is 80%. If an email arrives with the word "prize", what is the conditional probability that this email is spam? Use the Bayes theorem.

5. Given the sentence "it is rainng outside", you are tasked to normalize the text. Explain the steps and the final normalized text you'd get.

6. You’re training a trigram language model and your corpus includes the sentence: “A beautiful, breezy day”. What would the sequences look like under the context of your model? 

7. Your task is to classify tweets into either "positive", "negative", or "neutral". You have chosen to use a Feedforward Neural Network. Explain the features you would use and why.

8. Given the sentence "I am a student from XYZ University", identify the entities in the sentence and label them using Named Entity Recognition.

9. Explain an instance where an "Encoder-Decoder" model in Sequence-to-Sequence models would be more effective than a "Attention" model and why.

10. You’re using the Transformer model for English to French translation. The sentence is “I am going to the market”. Explain the role played by self-attention in accurately translating this sentence.

### Question - Next time, include solutions:
Of course, I apologize for that. Here are the solutions to the questions:

1. Using Finite State Automata, we consider the dictionary of known words and grammatical rules. As 'chase' is incorrect grammatically in this sentence, the model corrects it to 'chased' to form "The dog chased the cat".

2. "Unhappiness" can be broken down into three morphemes: the prefix "un-", the root "happy", and the suffix "-ness". "Un-" is a morpheme that adds the meaning of negation, "happy" is the root word implying a feeling of joy, and "-ness" transforms an adjective into a noun, indicating a state or condition.

3. Building a finite-state grammar to recognize email addresses with Pynini would involve defining the structure of a typical email address (one or more alphanumeric characters, followed by "@", followed by one or more alphanumeric characters, then a period, then a domain).

4. Using Bayes' theorem, the conditional probability of an email being spam given it contains the word "prize" is: P(Spam | Prize) = [P(Prize | Spam) * P(Spam)] / P(Prize) = [0.80*0.30] / P(Prize) = 0.24 / P(Prize). You would need the probability of any email containing the word "prize" to get the exact answer.

5. Normalizing "it is rainng outside" would first involve correcting spelling to "it is raining outside". Further normalization might involve lowercasing (already done here) or removing punctuation (not applicable here). 

6. In a trigram model, given the sentence “A beautiful, breezy day”, the sequences would be: [“<s>”, “a”, “beautiful”], [“a”, “beautiful”, “breezy”], [“beautiful”, “breezy”, “day”], [“breezy”, “day”, “</s>”], where "<s>" and "</s>" are special tokens indicating the start and end of the sentence.

7. For classifying tweets, features to use with a Feedforward Neural Network include word frequency counts, presence or absence of certain words, length of tweet, the sentiment score of the tweet (which can be calculated using a sentiment lexicon), and the use of hashtags, mentions, and emoticons.

8. "I" would be the entity indicating the person making the statement, so it would be labeled as "PERSON". "XYZ University" would be labeled as "ORGANIZATION", as it's named entity representing an institution.

9. An instance where an "Encoder-Decoder" model would be more effective than an "Attention" model is when translating a language with high word alignment like French to English. This is because the "Encoder-Decoder" model already has a good performance with high word alignment. However, for languages with lower word alignment, the "Attention" model would fare better as it allows the model to focus on specific parts of the input sequence.

10. With the Transformer model, the sentence “I am going to the market” would be processed in parallel. The self-attention mechanism would create a weighted representation of each word in relation to all other words in the sentence, to capture the dependencies among the words, which helps produce a more accurate translation.
