import json
import random
import os
from datetime import datetime, timedelta

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class CalendarService:
    @staticmethod
    def setup(calendar_path: str, num_events: int, index_path: str):
        generation_result = CalendarService.generate_events(calendar_path, num_events, index_path)
        return generation_result

    @staticmethod
    def generate_events(calendar_path: str, num_events: int, index_path: str):
        """
        Generates a list of random calendar events, saves them to a JSON file,
        and creates a FAISS vector index from the events.

        This function creates synthetic calendar event data, including random titles,
        locations, times, and attendees. It then saves these events to a specified JSON file,
        loads the file using LangChain's JSONLoader, embeds the events using HuggingFace embeddings,
        and stores the resulting vectors in a FAISS index.

        Args:
            calendar_path (str): Path to write the generated calendar events JSON file.
            num_events (int): Maximum number of events to generate (actual number may vary).
            index_path (str): Directory path to save the generated FAISS index.

        Returns:
            str: A message indicating that the FAISS index was created.

        Side Effects:
            - Writes a new JSON file containing generated calendar events.
            - Creates and saves a FAISS index to disk.
        """
        if os.path.exists(calendar_path):
            if not os.path.exists(index_path):
                return CalendarService.create_calendar_vector_index(calendar_path, index_path, new_index=True)
            return "Index already exists"

        events = []
        titles = ["Meeting", "Workshop", "Conference", "Lunch", "Project Review", "Lab", "Lecture"]
        locations = ["Office", "Zoom", "Conference Hall", "Cafe", "Online"]

        for _ in range(random.randint(5, num_events)):
            start_time = datetime.now() + timedelta(days=random.randint(1, 30), hours=random.randint(8, 18))
            end_time = start_time + timedelta(hours=random.randint(1, 3))

            event = {
                "title": random.choice(titles),
                "location": random.choice(locations),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "attendees": [f"user{random.randint(1, 10)}@example.com" for _ in range(random.randint(1, 5))]
            }
            events.append(event)

        with open(calendar_path, "w") as json_file:
            json.dump(events, json_file, indent=4)

        return CalendarService.create_calendar_vector_index(calendar_path, index_path, new_index=True)

    @staticmethod
    def create_calendar_vector_index(file_path: str, index_path: str, new_index: bool = False):
        """
        Loads calendar events from a JSON file, creates a FAISS vector store,
        and saves the index locally.

        Args:
            file_path (str): Path to the JSON file containing events.
            index_path (str): Directory path to save the FAISS index.
            new_index (bool): Flag that lets us override existing index
        Returns:
            str: A message indicating that the FAISS index was created or not
        """

        if not new_index and os.path.exists(index_path):
            return "Index already exists"

        # Load documents
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".[]",
            text_content=False
        )
        docs = loader.load()

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)

        # Save the index locally
        vector_store.save_local(index_path)

        return "Index created"
