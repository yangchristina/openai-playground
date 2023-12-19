"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const llm_api_1 = require("llm-api");
require("dotenv/config");
const fs_1 = require("fs");
const prompts_1 = require("@inquirer/prompts");
const SYSTEM_MESSAGE = "I am a student who is trying to study for my final exam. You are my tutor, and I will ask you questions about the topics I am studying. I will also ask you to give me a brief overview of the topics I am studying and how the topics I am studying can be applied to different scenarios. The topics I am studying are: ";
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
];
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
];
const init = () => __awaiter(void 0, void 0, void 0, function* () {
    const modelOptions = {
        openai: new llm_api_1.OpenAIChatApi({ apiKey: process.env.GPT_API_KEY }, { model: 'gpt-4-0613', contextSize: 8129 }),
    };
    console.log("welcome to my study helper");
    console.log("What topics would you like to study?");
    console.log("Loading");
    let { history, overview } = yield briefOverview(modelOptions.openai, EXAMPLE_TOPICS[0]);
    console.log("All set!");
    let content = overview + '\n';
    let scenarioCount = 0;
    let quizCount = 0;
    let testCount = 0;
    while (true) {
        const answer = yield (0, prompts_1.select)({
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
                    name: 'write',
                    value: 'write',
                    description: 'write to file',
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
            console.log(overview);
        }
        else if (answer === "scenario") {
            scenarioCount++;
            const answer = yield (0, prompts_1.input)({ message: "What is the scenario?" });
            const res = yield applyTopicsToScenario(modelOptions.openai, history, answer);
            history = res.history;
            content += `\n### Scenario #${scenarioCount}:\n` + res.content + '\n';
            quizOneTopic;
        }
        else if (answer === "quiz") {
            quizCount++;
            const res = yield quizOneTopic(modelOptions.openai, history);
            history = res.history;
            content += `\n### Quiz #${quizCount}:\n` + res.content + '\n';
        }
        else if (answer === "history") {
            console.log(history);
        }
        else if (answer === "test") {
            testCount++;
            const res = yield testAll(modelOptions.openai, history);
            history = res.history;
            content += `\n### Test #${testCount}:\n` + res.content + '\n';
        }
        else if (answer === "write") {
            (0, fs_1.writeFileSync)('response.md', content || "No content");
        }
        else if (answer === "question") {
            const answer = yield (0, prompts_1.input)({ message: "What is the question?" });
            const res = yield handleQuestion(modelOptions.openai, history, answer);
            history = res.history;
            content += `\n### Question - ${answer}:\n` + res.content + '\n';
        }
        else if (answer === "exit") {
            (0, fs_1.writeFileSync)('response.md', content || "No content");
            break;
        }
    }
});
const briefOverview = (openai, topics) => __awaiter(void 0, void 0, void 0, function* () {
    const overview = "Give me a brief overview on these topics: " + topics;
    try {
        const res = yield openai.textCompletion(overview, { systemMessage: SYSTEM_MESSAGE });
        console.log(res.content);
        return { history: [res.message], overview: res.content };
    }
    catch (e) {
        if (e instanceof llm_api_1.TokenError) {
            // handle token errors...
            console.error('tokenerr', e);
        }
        throw e;
    }
});
const applyTopicsToScenario = (openai, history, prompt) => __awaiter(void 0, void 0, void 0, function* () {
    const start = "How can I apply these topics to the following scenario. Answer in a way that helps me understand these topics and concepts better. Scenario:";
    return yield handleQuestion(openai, history, start + prompt);
});
const quizOneTopic = (openai, history) => __awaiter(void 0, void 0, void 0, function* () {
    const answer = yield (0, prompts_1.input)({ message: "What is the topic?" });
    const prompt = "Write me a short 3 question, open ended quiz on the following topic. Answer in a way that helps me understand these topics and concepts better. Topic: " + answer;
    return yield handleQuestion(openai, history, prompt);
});
const testAll = (openai, history) => __awaiter(void 0, void 0, void 0, function* () {
    const prompt = "Write me a 10 question test on the material. The test should make me apply my knowledge to scenarios. Here are some example questions:\n";
    return yield handleQuestion(openai, history, prompt + EXAMPLE_QUESTIONS.join('\n\n'));
});
init();
function handleQuestion(openai, history, prompt) {
    return __awaiter(this, void 0, void 0, function* () {
        try {
            const res = yield openai.chatCompletion([...history, { role: 'user', content: prompt }], {});
            console.log(res.content);
            return { history: [...history, res.message], content: res.content };
        }
        catch (e) {
            if (e instanceof llm_api_1.TokenError) {
                // handle token errors...
                console.error('tokenerr', e);
            }
            throw e;
        }
    });
}
