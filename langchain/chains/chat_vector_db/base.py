"""Chain for chatting with a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.vectorstores.base import VectorStore
import tiktoken
enc = tiktoken.get_encoding("gpt2")
MAX_ALLOWED_TOKEN = 3500


def _get_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class ChatVectorDBChain(Chain, BaseModel):
    """Chain for chatting with a vector database."""

    vectorstore: VectorStore
    combine_docs_chains: list[BaseCombineDocumentsChain]
    question_generator: LLMChain
    output_key: str = "answer"
    return_source_documents: bool = False
    top_k_docs_for_context: int = 4
    """Return the source documents."""

    @property
    def _chain_type(self) -> str:
        return "chat-vector-db"

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        qa_prompt: BasePromptTemplate = QA_PROMPT,
        **kwargs: Any,
    ) -> ChatVectorDBChain:
        ## This is the default chain type
        """Load chain from LLM."""
        stuff_chain = load_qa_chain(
            llm,
            chain_type="stuff",
            prompt=qa_prompt,
        )
        map_reduce_chain = load_qa_chain(
            llm,
            chain_type="map_reduce",
            prompt=qa_prompt,
        )
        refine_chain = load_qa_chain(
            llm,
            chain_type="refine",
            prompt=qa_prompt,
        )

        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)
        return cls(
            vectorstore=vectorstore,
            combine_docs_chains=[stuff_chain, map_reduce_chain, refine_chain],
            question_generator=condense_question_chain,
            **kwargs,
        )

    def count_tokens(self, docs):
        total_num_tokens = 0
        for doc in docs:
            total_num_tokens = total_num_tokens + len(enc.encode(doc.page_content))
        return total_num_tokens


    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        if chat_history_str:
            new_question = self.question_generator.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        docs = self.vectorstore.similarity_search(
            new_question, k=self.top_k_docs_for_context, **vectordbkwargs
        )
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str

        total_num_tokens = self.count_tokens(docs)
        if total_num_tokens < MAX_ALLOWED_TOKEN:
            print(f"Processing answer with {len(docs)} docs and {total_num_tokens} tokens, using stuff chain")
            # If the total number of tokens can be handled by one request, use stuff chain 
            answer, _ = self.combine_docs_chains[0].combine_docs(docs, **new_inputs)
        else:
            print(f"Processing answer with {len(docs)} docs and {total_num_tokens} tokens, using map_reduce chain")
            # Otherwise, use map-reduce chain
            answer, _ = self.combine_docs_chains[1].combine_docs(docs, **new_inputs)

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        if chat_history_str:
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        # TODO: This blocks the event loop, but it's not clear how to avoid it.
        docs = self.vectorstore.similarity_search(
            new_question, k=self.top_k_docs_for_context, **vectordbkwargs
        )
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = await self.combine_docs_chains[0].acombine_docs(docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}
