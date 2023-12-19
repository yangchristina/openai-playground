That's quite a comprehensive list you have there! All of these are integral to computational linguistics and natural language processing (NLP). Let's dive in.

1. **Finite State Text Processing, Morphology, Pynini** - Finite State Text Processing is using Finite State Automata (FSA) to process text. Morphology is the study of words, how they are formed and their relationship to other words. Pynini is a library that employs weighted finite-state transducer technology, thus enabling text transformation like normalization and casing adjustment.

2. **Text normalization & Spelling** - Text normalization involves transforming text into a more uniform format, which is essential for data preprocessing in NLP tasks. Spelling correction is a crucial subset of this and involves correcting spelling errors within the text.

3. **FSA, Reg Expressions, Conditional Prob., Bayes** - FSA is a mathematical model of computation, used for designing computer-based systems. Regular Expressions are sequences of characters that define a search pattern in text processing. Conditional probability is the probability of an event given another event has occurred; it’s fundamental to the working of Naïve Bayes Classifier.

4. **Language Models: Traditional vs Neural** - Traditional language models like n-grams predict the probability distribution of a sequence of words based on statistical methods. But neural language models like Recurrent Neural Networks (RNN), Long Short Term Memory (LSTM) use deep learning to predict word sequences, capturing semantic relationships better.

5. **Text Classification - Traditional Methods (Naive Bayes and Logistic Regression)** - Text classification involves classifying text into predefined categories. Naive Bayes uses the principles of Bayes theorem with an assumption of independence between features, while Logistic Regression calculates the probability of a certain class-assignment.

6. **Text Classification - Neural Methods (MLP and CNN)** – Multilayer perceptrons (MLP) are feedforward neural networks used for classification. Convolutional Neural Networks (CNN) are another type of deep learning model particularly useful for classifying images but can also be used for text data.

7. **Sequence labeling: Markov Models - POS tagging and NER, RNNs, LSTMs** - Sequence labeling involves assigning a categorical label to each member of the sequence of observed values. Markov Models, RNNs, and LSTM models, are all commonly used for this task. Part of Speech (POS) tagging and Named Entity Recognition (NER) are instances of sequence labeling.

8. **Sequence-to-Sequence: Encoder-Decoder, Attention** - Seq2seq models with encoder-decoder framework are used in NLP tasks like machine translation, text summarization. Attention mechanisms help the model to focus on different parts of the input when producing the output.

9. **Transformers and Text Classification (BOW, CBOW, transformers)** - Transformers are a type of deep learning model which handle long-term dependencies well. For text classification, Bag of Words (BOW), Continuous Bag of Words (CBOW) and transformer models like BERT can be used.

10. **Pre-trained language models** - These are language modeling models that were trained on a large corpus and can be fine-tuned on a specific task, reducing the need for a larger dataset.

11. **Intro to Syntax, Context Free Grammars and Parsing** - Syntax is the set of rules in a language that affects the order of words and phrases. Context-Free Grammars (CFG) are used to describe the syntax of programming languages. Parsing is the process of analyzing a string of symbols conforming to the rules of a formal grammar.

12. **Chunking, Dependency Parsing, Treebanks** - Chunking is a process of extracting phrases from unstructured text, Dependency Parsing analyzes the grammatical structure of a sentence, establishing relationships between 'head' words and words that modify those heads. Treebank is a database of sentences annotated with syntactic or semantic structures.

And so on. The fields of lexical semantics, topic modeling, sequence modeling and others each offer tools and techniques to deal with nuanced aspects of textual data. 

Most of these topics can be applied in various scenarios such as text classification, sentiment analysis, machine translation, speech recognition, information extraction, question-answering and more.

### Quiz #0:
Sure, let's formulate a few questions on morphology:

1. **What is morphology in the context of computational linguistics, and why is it important in natural language processing?**

   Answer: Morphology is a field within linguistics that studies the structure of words. In computational linguistics, this involves understanding how words are formed by analyzing their root forms (stems), prefixes, suffixes, and inflectional markers. This is important in natural language processing because it allows machines to understand and categorize words based on their significant parts, enhancing their capability to comprehend, process, and generate human language efficiently. For example, by understanding the root word "happy" and the suffix "-ness", a machine can understand that "happiness" refers to the state of being happy.

2. **Can you explain the difference between inflectional and derivational morphology? Provide an example of each.**

   Answer: Inflectional and derivational morphologies are two key sub-fields within morphology. Inflectional morphology deals with inflections that do not change a word's grammatical category (part of speech) nor create a new word but instead provide grammatical information such as tense or number. For example, adding "-s" to a verb to indicate the present tense third-person singular (as in "walks").

   On the other hand, derivational morphology involves the addition of affixes (prefixes, infixes, suffixes) to a root or stem word to create a new word, often changing the word's grammatical category. For example, adding "-ness" to the adjective "happy" forms the noun "happiness".

3.  **How does a sound understanding of morphology contribute to developing more accurate and efficient machine translation systems?**

   Answer: Morphology, being the study of word formation, allows machine translation systems to understand the structure of words, their origins, and how they can change, which is crucial when translating from one language to another. This is particularly relevant for languages with rich morphological structures where words can take on many forms. Recognizing the root word and various affixes can greatly improve the accuracy of translation. For instance, distinguishing between the singular form of a noun and its plural form can impact the overall meaning of a sentence, hence it’s important to translate them accurately. Thus, incorporating morphology can make machine translation systems more sophisticated and accurate, thereby enhancing their performance.
