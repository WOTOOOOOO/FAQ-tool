# University Regulations, Calendar & Student Data Query Tool

## Overview

A simple LLM tool that analyzes provided university regulations, student, and calendar information to help answer your questions.

## Features

- **Analyze Regulations**: The app takes in your regulations file, splits it up logically, and uses it to answer your questions.
- **Analyze Student Information**: The app takes your student info as a CSV file and helps you extract various kinds of information from it. You won't be allowed to modify the file, as that would be insecure.
- **Analyze Calendar**: The app takes in your calendar information as a JSON file and answers questions regarding events.

## Installation

1. Clone the repository to your local machine:
   ```
   git clone git@github.com:WOTOOOOOO/FAQ-tool.git
   cd <code directory>

2. Install requirements:
   ```
   pip install -r requirements.txt

3. Add your Groq GROQ_API_KEY to a .env file.

## How to Run

1. Open a terminal in the project directory.
2. Run the app using the Streamlit command:
   ```
   streamlit run app.py

## Usage Instructions

1. Enter your query in the text box.
2. Press Submit.
3. Wait for the magic!

## Application Workflow

1. Accept query.
2. Decide which tool to use (if any).
3. Call the tool.
4. Process tool response.
5. Possible HITL response
6. Return the final response.

## Additional Notes

1. The application generates its own student CSV file, as actual student data is unavailable.
2. It also generates calendar events, since Microsoft Calendar API does not function reliably, and even if it did, its usability cannot be verified.

## Limitations

For student data queries, request answers that will be succinct, as the free version of the LLM model has token limits.

