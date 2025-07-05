from pydantic import BaseModel
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.agents import AgentFinish, AgentAction


class GraphState(BaseModel):
    query: str
    retrieved: Optional[list[Document]] = None
    intermediate_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    context: Optional[str] = None
    hitl_ui_needed: Optional[str] = None
    code_review_needed: Optional[str] = None
    selected_tool: Optional[str] = None


class UniversityGraph:
    def __init__(self, university_agent):
        self.university_agent = university_agent
        self.graph = None

    def decide_route(self, query):
        plan = self.university_agent.agent.agent.plan(
            input=query,
            chat_history=self.university_agent.memory.chat_memory.messages,
            intermediate_steps=[]
        )

        if isinstance(plan, AgentFinish):
            return "None", "default_node"

        tool_name = plan.tool

        if tool_name == "Calendar events retriever":
            return tool_name, "calendar_node"
        if tool_name == "Current Date and Time tool":
            return tool_name, "date_time_node"
        if tool_name == "Pandas student Data Frame Tool":
            return tool_name, "pandas_node"
        if tool_name == "University regulations retriever":
            return tool_name, "retrieve"

        return tool_name, "default_node"

    def default_node(self, state: GraphState) -> GraphState:
        guided_query = (
            f"Without using any tools respond to this query {state.query}"
        )

        output = self.university_agent.agent.invoke({"input": guided_query})
        return state.model_copy(update={
            "output": output.get("output", output),
        })

    def decision_routing_node(self, state: GraphState) -> GraphState:
        tool, route = self.decide_route(state.query)
        return state.model_copy(update={
            "metadata": {"next_node": route},
            "selected_tool": tool
        })

    def regulations_rag_retrieval_node(self, state: GraphState) -> GraphState:
        query = state.query
        docs_with_scores = self.university_agent.vector_store_regulations.similarity_search_with_score(query, k=3)
        retrieved_docs = [doc for doc, score in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        max_possible_distance = 2.0
        normalized_scores = [max(0.0, 1 - (s / max_possible_distance)) for s in scores]
        confidence = round(sum(normalized_scores) / len(normalized_scores), 3)

        return state.model_copy(update={
            "retrieved": retrieved_docs,
            "intermediate_response": "Documents retrieved based on FAISS search.",
            "metadata": {"confidence": confidence}
        })

    def rag_answer_generation_node(self, state: GraphState) -> GraphState:
        context = "\n\n".join([doc.page_content for doc in state.retrieved])
        tool = state.selected_tool or "a relevant tool"

        guided_query = (
            f"Use the tool '{tool}' to answer this question. "
            f"Now, answer the user query:\n{state.query}"
        )

        try:
            output = self.university_agent.agent.invoke({"input": guided_query})
            return state.model_copy(update={
                "output": output.get("output", output),
                "context": context
            })
        except Exception as e:
            return state.model_copy(update={
                "output": f"Agent failed while trying to use {tool}: {str(e)}",
                "context": context
            })

    def calendar_rag_retrieval_node(self, state: GraphState) -> GraphState:
        query = state.query
        docs_with_scores = self.university_agent.vector_store_calendar.similarity_search_with_score(query, k=20)
        retrieved_docs = [doc for doc, score in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        max_possible_distance = 2.0
        normalized_scores = [max(0.0, 1 - (s / max_possible_distance)) for s in scores]
        confidence = round(sum(normalized_scores) / len(normalized_scores), 3)

        return state.model_copy(update={
            "retrieved": retrieved_docs,
            "intermediate_response": "Documents retrieved based on FAISS search.",
            "metadata": {"confidence": confidence}
        })

    def rag_final(self, state: GraphState) -> GraphState:
        if state.metadata.get("confidence", 1.0) < 0.8:
            return state.model_copy(update={"hitl_ui_needed": True})
        return state.model_copy()

    def pandas_code_review(self, state: GraphState) -> GraphState:
        query = state.query

        # Run the agent and capture intermediate steps
        result = self.university_agent.pandas_agent.invoke({"input": query})
        actions = result.get("intermediate_steps", [])
        code = actions[-1][0].tool_input

        answer = result.get("output", "")

        return state.model_copy(
            update={
                "context": answer,
                "output": code,
                "code_review_needed": True
            }
        )

    def pandas_execute_user_code(self, state: GraphState) -> GraphState:
        code_as_text = state.query
        try:
            query = f"Execute the following code {code_as_text}"
            result = self.university_agent.pandas_agent.invoke({"input": query})

            answer = result.get("output", "")
            return state.model_copy(update={"output": answer})

        except Exception as e:
            return state.model_copy(update={
                "output": f"Agent failed to execute code via Pandas agent: {e}"
            })

    def generate_code_response(self, code: str) -> Dict[str, Any]:
        query = f"execute this code on the dataframe: {code}"
        result_state = self.pandas_execute_user_code(GraphState(query=query))
        return result_state.model_dump()

    def generate_response(self, query: str) -> Dict[str, Any]:
        result_state = self.graph.invoke(GraphState(query=query))
        return result_state

    def build_langgraph(self):
        builder = StateGraph(GraphState)

        builder.add_node("route", RunnableLambda(self.decision_routing_node))
        builder.add_node("default_node", RunnableLambda(self.default_node))
        builder.add_node("regulations_retrieve", RunnableLambda(self.regulations_rag_retrieval_node))
        builder.add_node("calendar_retrieve", RunnableLambda(self.calendar_rag_retrieval_node))
        builder.add_node("generate", RunnableLambda(self.rag_answer_generation_node))
        builder.add_node("rag_final", RunnableLambda(self.rag_final))
        builder.add_node("pandas_code_review", RunnableLambda(self.pandas_code_review))

        builder.set_entry_point("route")

        builder.add_conditional_edges(
            "route",
            lambda state: state.metadata["next_node"],
            {
                "retrieve": "regulations_retrieve",
                "pandas_node": "pandas_code_review",
                "calendar_node": "calendar_retrieve",
                "date_time_node": END,
                "default_node": "default_node",
                None: END
            }
        )

        # Default path
        builder.add_edge("default_node", END)

        # University RAG path
        builder.add_edge("regulations_retrieve", "generate")
        builder.add_edge("generate", "rag_final")
        builder.add_edge("rag_final", END)

        # Calendar RAG path
        builder.add_edge("calendar_retrieve", "generate")
        builder.add_edge("generate", "rag_final")
        builder.add_edge("rag_final", END)

        # Pandas path
        builder.add_edge("pandas_code_review", END)

        self.graph = builder.compile()
