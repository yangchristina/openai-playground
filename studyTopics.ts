import { ChatRequestMessage, OpenAIChatApi, TokenError } from 'llm-api';
import 'dotenv/config'
import { writeFileSync } from 'fs';
import { input, select } from '@inquirer/prompts';

const textbook = "https://web.stanford.edu/~jurafsky/slp3/"
const SYSTEM_MESSAGE = "I am a student who is trying to study for my final exam. You are my tutor, and I will ask you questions about the topics I am studying. I will also ask you to give me a brief overview of the topics I am studying and how the topics I am studying can be applied to different scenarios. The topics I am studying are: "
const EXAMPLE_TOPICS = [
    `Finite State Text Processing, Morphology, Pynini

    Text normalization, Spelling

    FSA, Reg Expressions, Conditional Prob., Bayes

    Text Normalization, Pynini and Spell Checking

    Language Models: Traditional vs Neural

    Text Classification - Traditional Methods (Naive Bayes and Logistic Regression)

    Feedforward Neural NetworksLinks to an external site. (MLP)

    Text Classification - Neural Methods (MLP and CNN)

    Sequence labeling: Markov Models - POS tagging and NER
    Language Models

    Sequence labeling: RNNs, LSTMs

    Sequence-to-Sequence: Encoder-Decoder, Attention

    Transformers

    Text Classification (BOW, CBOW, transformers).

    Pre-trained language models

    Intro to syntax, Context Free Grammars and Parsing

    Chunking, Dependency Parsing, Treebanks

    Sequence Modeling

    Dependency/Constituency Parsing PCFG Traditional CKY + Neural Models

    Intro Semantics

    Semantic Role Labeling

    Lexical Semantics

    Syntax

    Topic Modeling

    word embeddings

    Discourse and Coreference

    Summarization`
]
const EXAMPLE_QUESTIONS = [
    `Question: Consider the sentence “She wants to stay as far away from them as humanly
    possible”. Which of the following language models: count-based trigram, neural trigram, or
    RNN-based model is likely to give it the highest probability and why?

    Answer: The RNN will give it the highest probability because of the long-range dependency between
    the first “as” and the second “as”. Both trigram models (whether implemented as a
    count-based or a neural model) only look at a window of 2 context words when computing
    the next token probability, and they will “forget” about the first “as” by the time they need to
    predict the second, because they are 5 words apart. Conversely, the RNN doesn’t follow the
    Markov assumption, and the hidden state at each time step captures the entire history.`,
    `Question: Assume the following sentences were generated by a language model given the
    prompt “Can you help me with something?”:
    i. I don’t know I don’t know
    ii. Exoskeletal junction at the railroad delayed
    iii. Sure, what can I do for you?
    Take a guess which of the following decoding strategies generated each sentence:
    sampling from the entire distribution, greedy decoding, or top k with a relatively small k.
    Explain your answer. In particular, focus on how the different strategies work and the
    tradeoffs that different decoding methods introduce.

    Answer: Sentence 1 was generated by greedy decoding (which is equivalent to top k with k = 1). This
    strategy predicts the most likely word at each timestep, so it typically generates boring and
    repetitive sentences with very common words.
    Sentence 2 was generated by sampling from the entire distribution (which is equivalent to
    top k with k = |V|). This strategy randomly samples a word from the vocabulary
    proportionally to its probability in the next token distribution. While this strategy leads to
    more diversity in the generated sentences than greedy decoding, it puts too much of the
    probability mass on the “long tail” of the distribution and may generate rare words.
    Sentence 3 was generated by top k with a relatively small k. Top k is a compromise between
    greedy decoding and sampling from the entire distribution: It first prunes the distribution to
    the most likely k words, then renormalizes it and samples a word from the pruned
    distribution proportionally to its probability. It yields higher quality and moderately diverse
    sentences (depending on the value of k).`,
    `Question: What is the advantage of bidirectional RNNs over unidirectional RNNs? Describe how the hidden states are computed using a bidirectional RNN.`,
    `Question: In decoding, let V be the size of vocabulary, and M be the maximum output length. What are the respective time and space complexity of beam search (beam size equals to K) and exhaustive search (i.e. exploring all sequences)?`,
    `Question: During beam search with beam size=K, what happens when an “end-of-sequence” token is generated and the corresponding sequence belongs to the K best hypothesis? Why?`,
]

const init = async () => {
    const modelOptions = {
        openai: new OpenAIChatApi(
            { apiKey: process.env.GPT_API_KEY },
            { model: 'gpt-4-0613', contextSize: 8129 },
        ),
    }
    console.log("welcome to my study helper")
    console.log("What topics would you like to study?")

    console.log("Loading")
    let { history, overview } = await briefOverview(modelOptions.openai, EXAMPLE_TOPICS[0])
    console.log("All set!")

    let content = overview + '\n'
    let scenarioCount = 0
    let quizCount = 0
    let testCount = 0
    while (true) {
        const answer = await select({
            message: 'What would you like to do?',
            choices: [
                {
                    name: 'overview',
                    value: 'overview',
                    description: 'See topic overviews',
                },
                {
                    name: 'scenario',
                    value: 'scenario',
                    description: 'apply scenario to topics',
                },
                {
                    name: 'history',
                    value: 'history',
                    description: 'see history',
                },
                {
                    name: '1 topic quiz',
                    value: 'quiz',
                },
                {
                    name: 'test all topics',
                    value: 'test',
                },
                {
                    name: 'question',
                    value: 'question',
                    description: 'ask a question',
                },
                {
                    name: 'exit',
                    value: 'exit',
                    description: 'exits the program',
                },
            ],
        });
        if (answer === "overview") {
            console.log(overview)
        } else if (answer === "scenario") {
            scenarioCount++
            const answer = await input({ message: "What is the scenario?" })
            const res = await applyTopicsToScenario(modelOptions.openai, history, answer)
            history = res.history
            content += `\n### Scenario #${scenarioCount}:\n` + res.content + '\n'
        } else if (answer === "quiz") {
            quizCount++
            const res = await quizOneTopic(modelOptions.openai, history)
            history = res.history
            content += `\n### Quiz #${quizCount}:\n` + res.content + '\n'
        } else if (answer === "history") {
            console.log(history)
        } else if (answer === "test") {
            testCount++
            const res = await testAll(modelOptions.openai, history)
            history = res.history
            content += `\n### Test #${testCount}:\n` + res.content + '\n'
        }
        else if (answer === "question") {
            const question = await input({ message: "What is the question?" })
            const res = await handleQuestion(modelOptions.openai, history, question)
            history = res.history
            content += `\n### Question - ${question}:\n` + res.content + '\n'
        }
        else if (answer === "exit") {
            break;
        }
        writeFileSync('response.md', content || "No content");
    }
}

const briefOverview = async (openai: OpenAIChatApi, topics: string) => {
    const overview = "Give me a brief overview on these topics: " + topics + '\n These topics and the material I am learning come from Speech and Language Processing (3rd ed. draft) by Dan Jurafsky and James H. Martin'

    try {
        const res = await openai.textCompletion(overview, { systemMessage: SYSTEM_MESSAGE });
        console.log(res.content)
        return { history: [res.message], overview: res.content }
    } catch (e) {
        if (e instanceof TokenError) {
            // handle token errors...
            console.error('tokenerr', e)
        }
        throw e
    }
}

const applyTopicsToScenario = async (openai: OpenAIChatApi, history: ChatRequestMessage[], prompt: string) => {
    const start = "How can I apply these topics to the following scenario. Answer in a way that helps me understand these topics and concepts better. Scenario:"

    return await handleQuestion(openai, history, start + prompt)
}


const quizOneTopic = async (openai: OpenAIChatApi, history: ChatRequestMessage[]) => {
    const answer = await input({ message: "What is the topic?" })
    const prompt = `Write me a short 3 question quiz on the following topic: ${answer}.  The questions should make me apply my knowledge to scenarios.`
    const examples = "Here are some example questions:\n" + EXAMPLE_QUESTIONS.join('\n\n')
    return await handleQuestion(openai, history, prompt + '\n\n' + examples)
}

const testAll = async (openai: OpenAIChatApi, history: ChatRequestMessage[]) => {
    const prompt = "Write me a 10 question test on the material. The test should make me apply my knowledge to scenarios. Include solutions. Here are some example questions:\n"
    return await handleQuestion(openai, history, prompt + EXAMPLE_QUESTIONS.join('\n\n'))
}

init()


async function handleQuestion(openai: OpenAIChatApi, history: ChatRequestMessage[], prompt: string) {
    try {
        const res = await openai.chatCompletion([...history, { role: 'user', content: prompt }], {});

        console.log(res.content)
        return { history: [...history, res.message], content: res.content }
    } catch (e) {
        if (e instanceof TokenError) {
            // handle token errors...
            console.error('tokenerr', e)
        }
        throw e
    }
}