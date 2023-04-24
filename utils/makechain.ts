import { OpenAIChat } from 'langchain/llms';
import { PineconeStore } from 'langchain/vectorstores';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are an AI companion for Magic The Gathering, you role is to help users to better understand Magic's rules. The document you have been trained on is the comprehensive Rules of Magic The Gathering,
it's a reference document that holds all of the rules and possible corner cases found in Magic The Gathering.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAIChat({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-3.5-turbo if you don't have access to gpt-4 api
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );

  return chain;
};
