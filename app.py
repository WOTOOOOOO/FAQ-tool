import sys

import streamlit as st

from constants import *
from regulations_service import RegulationService
from calendar_service import CalendarService
from main import UniversityQueryAgent
from student_generator_util import generate_student_data_csv

sys.modules['torch.classes'] = None


class QueryApp:
    def __init__(self):
        self.title = "University Regulations, Calendar & Student Data Query Tool"
        self.query_key = "query_input"
        self.chat_history_key = "chat_history"
        self.service_init_key = "services_initialized"
        self.student_data_key = "student_data_generated"
        self.faq_agent_key = "faq_agent_initialized"

    def setup(self):
        self._generate_students()
        self._initialize_services()
        self._initialize_chat_history()
        self._initialize_faq_agent(FAISS_REGULATIONS_INDEX_NAME, FAISS_CALENDAR_INDEX_NAME, STUDENTS_CSV)

    def run(self):
        self.setup()
        st.title(self.title)
        query = self._get_user_input()
        self._handle_query_submission(query)
        self._display_chat_history()

    def _initialize_faq_agent(self, faiss_regulations_index: str, faiss_calendar_index: str, students_csv: str):
        if self.faq_agent_key not in st.session_state:
            st.session_state[self.faq_agent_key] = UniversityQueryAgent(
                faiss_regulations_index,
                faiss_calendar_index,
                students_csv
            )

    def _generate_students(self):
        if self.student_data_key not in st.session_state:
            generate_student_data_csv(STUDENTS_CSV, NUMBER_OF_STUDENTS, COURSE_PRICE)
            st.session_state[self.student_data_key] = True

    def _initialize_services(self):
        if self.service_init_key not in st.session_state:
            CalendarService.setup(CALENDAR_EVENTS, NUMBER_OF_EVENTS, FAISS_CALENDAR_INDEX_NAME)
            RegulationService.setup(REGULATIONS_PDF, FAISS_REGULATIONS_INDEX_NAME, NUMBER_OF_REGULATIONS_CHUNKS)
            st.session_state[self.service_init_key] = True

    def _initialize_chat_history(self):
        if self.chat_history_key not in st.session_state:
            st.session_state[self.chat_history_key] = []

    def _get_user_input(self):
        return st.text_area("Enter your query:", key=self.query_key)

    def _handle_query_submission(self, query):
        if st.button("Submit") and query:
            response = self._generate_response(query)
            st.session_state[self.chat_history_key] = [{"query": query, "response": response}] + st.session_state[
                self.chat_history_key]

    def _generate_response(self, query):
        return st.session_state[self.faq_agent_key].generate_response(query)

    def _display_chat_history(self):
        st.write("### Chat History:")
        with st.container():
            for entry in st.session_state[self.chat_history_key]:
                st.markdown(f"**User:** {entry['query']}")
                st.markdown(f"**Assistant:** {entry['response']['output']}")
                st.write("---")


if __name__ == "__main__":
    app = QueryApp()
    app.run()
