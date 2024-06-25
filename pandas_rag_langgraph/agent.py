import re
from typing import Annotated, Iterator, Literal, TypedDict

from langchain import hub
from langchain_community.document_loaders import web_base
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from pandas_rag_langgraph.utils import RemoveMessage, add_messages


# Index 3 pages from Pandas user guides
SOURCE_URLS = [
    'https://pandas.pydata.org/docs/user_guide/indexing.html',
    'https://pandas.pydata.org/docs/user_guide/groupby.html',
    'https://pandas.pydata.org/docs/user_guide/merging.html'
]

NEWLINE_RE = re.compile("\n+")

class PandasDocsLoader(web_base.WebBaseLoader):
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            text = soup.get_text(**self.bs_get_text_kwargs)
            text = NEWLINE_RE.sub("\n", text)     
            metadata = web_base._build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)


def prepare_documents(urls: list[str]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"In \[[0-9]+\]",
            r"\n+",
            r"\s+"
        ],
        is_separator_regex=True,
        chunk_size=1000
    )
    docs = [PandasDocsLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return text_splitter.split_documents(docs_list)


def get_retriever() -> BaseRetriever:
    documents = prepare_documents(SOURCE_URLS)
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="pandas-rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    return retriever


# LLM / Retriever
llm = ChatOpenAI(model="gpt-4o", temperature=0)
retrieval_grader_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = get_retriever()


# Prompts / data models

RAG_PROMPT: ChatPromptTemplate = hub.pull("rlm/rag-prompt")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

RETRIEVAL_GRADER_SYSTEM = (
"""
You are a grader assessing relevance of a retrieved document to a user question.
Your goal is to filter out erroneous retrievals.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
)
RETRIEVAL_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RETRIEVAL_GRADER_SYSTEM),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


HALLUCINATION_GRADER_SYSTEM = (
"""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no', where 'yes' means that the answer is grounded in / supported by the set of facts.

IF the generation includes code examples, make sure those examples are FULLY present in the set of facts, otherwise always return score 'no'.
"""
)
HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_GRADER_SYSTEM),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


ANSWER_GRADER_SYSTEM = (
"""
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no', where 'yes' means that the answer resolves the question.
"""
)
ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_GRADER_SYSTEM),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)


QUERY_REWRITER_SYSTEM = (
"""
You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.
Look at the input and try to reason about the underlying semantic intent / meaning.
"""
)
QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_REWRITER_SYSTEM),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    retries: int


NOT_ENOUGH_INFORMATION = "Not enough information to answer"


def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = convert_to_messages(state["messages"])[-1].content

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state: GraphState, config):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", 3)

    if retries < max_retries:
        rag_chain = RAG_PROMPT | llm
        generation = rag_chain.invoke({"context": documents, "question": question})
    else:
        generation = AIMessage(content=NOT_ENOUGH_INFORMATION)

    state_update = {"retries": retries + 1}
    # make sure to remove previous AI message if we're regenerating
    last_message = convert_to_messages(state["messages"])[-1]
    if isinstance(last_message, AIMessage):
        state_update["messages"] = [RemoveMessage(id=last_message.id), generation]
    else:
        state_update["messages"] = [generation]
    return state_update


def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    retrieval_grader = RETRIEVAL_GRADER_PROMPT | retrieval_grader_llm.with_structured_output(GradeDocuments)

    # Score each doc
    filtered_docs = []
    grades: list[GradeDocuments] = retrieval_grader.batch([
        {"question": question, "document": d.page_content}
        for d in documents
    ])
    filtered_docs = [doc for doc, grade in zip(documents, grades) if grade.binary_score == "yes"]
    print(f"---GRADE: {len(filtered_docs)}/{len(documents)} RELEVANT DOCUMENTS---")
    return {"documents": filtered_docs}


def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]

    retries = state["retries"] if state.get("retries") is not None else -1

    # Re-write question
    query_rewriter = QUERY_REWRITER_PROMPT | llm | StrOutputParser()
    better_question = query_rewriter.invoke({"question": question})
    return {"question": better_question, "retries": retries + 1}


### Edges


def decide_to_generate(state: GraphState, config) -> Literal["transform_query", "generate"]:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", 3)

    if not filtered_documents and retries < max_retries:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents or we ran out of retries, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> Literal["not enough information", "not grounded", "useful", "not useful"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = convert_to_messages(state["messages"])[-1]

    if generation.content == NOT_ENOUGH_INFORMATION:
        return "not enough information"

    print("---CHECK HALLUCINATIONS---")
    hallucination_grader = HALLUCINATION_GRADER_PROMPT | llm.with_structured_output(GradeHallucinations)
    hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation.content}
    )

    # Check hallucination
    if hallucination_grade.binary_score != "yes":
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not grounded"

    print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

    # Check question-answering
    print("---GRADE GENERATION vs QUESTION---")

    answer_grader = ANSWER_GRADER_PROMPT | llm.with_structured_output(GradeAnswer)
    answer_grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation.content})
    if answer_grade.binary_score == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"


# Define graph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate)
workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not grounded": "generate",
        "not useful": "transform_query",
        "useful": END,
        "not enough information": END
    },
)

# Compile
graph = workflow.compile()