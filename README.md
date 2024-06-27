# Pandas RAG with LangGraph

This is a demo app for a RAG system that answers questions about [Pandas](https://pandas.pydata.org/) documentation powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain).

## Data

This demo uses a **very** small subset of [Pandas](https://pandas.pydata.org/) documentation for question answering, namely the following 3 pages from user guides:

* [Indexing](https://pandas.pydata.org/docs/user_guide/indexing.html)
* [GroupBy](https://pandas.pydata.org/docs/user_guide/groupby.html)
* [Merging](https://pandas.pydata.org/docs/user_guide/merging.html)

Such a small subset is chosen on purpose to demonstrate how the agent architecture below can flexibly handle model hallucinations. Specifically, there is a lot of knowledge about popular open-source libraries already present in the model weights. Therefore, if the model fails to look up data from vectorstore, it might still attempt to answer the question. In certain cases this behavior might be not desireable, and this demo demonstrates how you can handle those cases.

## Components

The demo uses the following components:

- LLM: Anthropic's Claude 3.5 Sonnet (`claude-3-5-sonnet-20240620`) via LangChain's [`ChatAnthropic`](https://python.langchain.com/v0.2/docs/integrations/chat/anthropic/). Specifically, the LLM is used for three different tasks:
  - candidate answer generation
  - grading answer hallucinations
  - grading answer relevance
- Embeddings: OpenAI Embeddings (`text-embedding-ada-002`) via LangChain's [`OpenAIEmbeddings`](https://python.langchain.com/v0.2/docs/integrations/text_embedding/openai/)
- Vectorstore: [Chroma DB](https://www.trychroma.com/) (via LangChain's [`Chroma`](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/)
  - **NOTE**: for production use cases you would likely need to deploy your own vector database instance or use a hosted solution
- Web search: [Tavily Search](https://tavily.com/) (via LangChain's [`TavilySearchResults`](https://python.langchain.com/v0.2/docs/integrations/tools/tavily_search/))

## Architecture

This demo implements a custom RAG architecture that combines ideas from [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884). For simplicity, it omits the document relevance grading step, but you can find full implementation of those papers in LangGraph [here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/) and [here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/).

The flow is as follows:

1. Search vectorstore for documents relevant to the user question
2. Generate a candidate answer based on those documents
3. Grade the candidate answer for hallucinations: is it grounded in the documents or did the model hallucinate?
  - if grounded, proceed to the next step (grading answer relevance)
  - if hallucination, re-generate the answer (return to step 2). Attempt re-generation at most N times (user-controlled)
4. Grade the candidate answer for relevance: did it address user's question?
  - if yes, return the candidate answer to the user
  - if no, rewrite the query and attempt the search (return to step 1) again. Attempt re-writing at most N times (user-controlled)
5. (Optional) If the answer is not grounded in the documents and/or not relevant after N tries, pass the user question to web search
6. (Optional) Generate an answer based on the web search results and return it to the user

See this flow chart for a visual representation.

![rag-agent](/static/pandas-rag-agent.png)

## Interacting with the graph

First, make sure that your environment variables are set in `.env` file. See `.env.example`.

```python
from dotenv import load_dotenv

load_dotenv()

from pandas_rag_langgraph.agent import graph

inputs = {"messages": [("human", "how do i calculate sum by groups")]}
for output in graph.stream(inputs):
    print(output)
```

## Deploying LangGraph applications

If you'd like to deploy a LangGraph application like the one in this demo, you can use [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/)

## Further work

This demo can be improved in a variety of ways:
- add chat history to ask follow up questions (see example of that in [Chat LangChain](https://github.com/langchain-ai/chat-langchain))
- add better document chunking
- add document relevance grader ([here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/) and [here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/))
- add a mixed keyword search ([BM25](https://en.wikipedia.org/wiki/Okapi_BM25)) & vector search solution for document retrieval