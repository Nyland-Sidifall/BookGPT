import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from "dotenv";
import readline from "readline";

// Load environment variables from .env file
dotenv.config();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let Question = "Explain vector spaces using python programming.";

const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 0,
});

const directoryLoader = new DirectoryLoader("Books", {
  ".pdf": (path) => new PDFLoader(path),
});

const docs = await directoryLoader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const splitDocs = await textSplitter.splitDocuments(docs);

const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  new OpenAIEmbeddings()
);

const retriever = vectorstore.asRetriever();

const systemTemplate = [
  `You are an assistant for question-answering tasks. `,
  `You are also an expert programmer who specifies in Python Programming. `,
  `Use the following pieces of retrieved context to answer the question.`,
  `If you don't know the answer, say that you don't know.`,
  `\n\n`,
  `{context}`,
].join("");

const prompt = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["human", "{input}"],
]);

const questionAnswerChain = await createStuffDocumentsChain({ model, prompt });
const ragChain = await createRetrievalChain({
  retriever,
  combineDocsChain: questionAnswerChain,
});

const results = await ragChain.invoke({
  input: Question,
});

console.log(results.answer);
console.log("Press ctrl + c to enter new question.");
