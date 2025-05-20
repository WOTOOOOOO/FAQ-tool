import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

from langchain.agents import AgentType, initialize_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.tools import Tool
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


class UniversityQueryAgent:
    def __init__(self, faiss_regulations_index: str, faiss_calendar_index: str, students_csv: str):
        load_dotenv()

        self.llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

        # Load student data
        self.df = pd.read_csv(students_csv)
        self.column_names = self.df.columns.tolist()

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Load regulation retriever
        self.vector_store_regulations = FAISS.load_local(
            faiss_regulations_index,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retrieval_chain_regulations = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store_regulations.as_retriever(search_kwargs={"k": 3}),
            chain_type="refine"
        )
        self.regulation_tool = Tool(
            name="University regulations retriever",
            func=lambda query: self.retrieval_chain_regulations.invoke(query),
            description="Retrieves and synthesizes information about university's academic regulations and guidelines."
        )

        # Load calendar retriever
        self.vector_store_calendar = FAISS.load_local(
            faiss_calendar_index,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retrieval_chain_calendar = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store_calendar.as_retriever(search_kwargs={"k": 20}),
            return_source_documents=True
        )
        self.calendar_tool = Tool(
            name="Calendar events retriever",
            func=lambda query: self.retrieval_chain_calendar.invoke(query),
            description="Retrieves information about upcoming calender events"
        )

        # Create Pandas agent with protection
        self.pandas_agent = create_pandas_dataframe_agent(
            self.llm, self.df, verbose=True, allow_dangerous_code=True
        )
        self.classification_prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                f"Our pandas DataFrame has the following columns: {self.column_names}.\n"
                "Determine if the following query entails the modification of the DataFrame:\n"
                "{query}\n"
                "Respond with 'SAFE' if it only reads data, otherwise respond with 'MODIFY'."
            )
        )
        self.classifier = self.classification_prompt | self.llm | StrOutputParser()

        self.pandas_tool = Tool(
            name="Pandas student Data Frame Tool",
            func=self.safe_query_execution,
            description="Use this tool to query structured student information."
        )

        # Current date and time tool
        self.datetime_tool = Tool(
            name="Current Date and Time tool",
            func=self.get_current_datetime,
            description="Fetches the current date and time."
        )

        # Memory and agent setup
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent = initialize_agent(
            tools=[self.regulation_tool, self.calendar_tool, self.pandas_tool, self.datetime_tool],
            llm=self.llm,
            memory=self.memory,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def safe_query_execution(self, query: str):
        """
        Executes a query only if classified as 'SAFE'. Prevents modifications.

        Args:
            query (str): The query to be classified and executed.

        Returns:
            Any: Query result if allowed, otherwise a warning message.
        """
        classification_result = self.classifier.invoke({"query": query})
        if "SAFE" in classification_result:
            return self.pandas_agent.invoke({"input": query})
        return "Modification blocked: This query is not allowed. Only read-only queries are permitted."

    @staticmethod
    def get_current_datetime(query: str):
        """
        Retrieves the current date and time in 'YYYY-MM-DD HH:MM:SS' format.

        Returns:
            str: The current timestamp as a formatted string.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_response(self, query: str):
        """
        Generates a response based on the input query using an agent.

        Args:
            query (str): The query to process.

        Returns:
            Any: The agent's response to the query or a default message on failure.
        """
        try:
            return self.agent.invoke({"input": query})
        except Exception as e:
            return {
                'output': f'An error occurred while processing the query: {str(e)}'
            }
