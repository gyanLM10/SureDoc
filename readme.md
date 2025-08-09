# Multi-Agent Doctor Appointment Booking System (LangGraph + Gemini 1.5 Flash)

## Overview
This project is a **Multi-Agent Doctor Appointment Booking System** built with:
- **LangGraph** – for orchestrating multiple conversational agents (nodes)
- **Google Gemini 1.5 Flash** – for fast, high-quality LLM responses
- **FastAPI** – for exposing an API endpoint
- **LangChain** – for LLM integration and state management

The agent network can:
- Understand natural language booking requests
- Ask clarifying questions if essential information is missing
- Handle both **booking** and **information gathering** flows
- Dynamically switch between conversation nodes

---

## Project Structure

├── data/
│ └── doctor_availability.csv 
├── data_models/
│ ├── init.py
│ └── models.py # Data model definitions
├── notebook/
│ ├── availability.csv 
│ └── multiagent_system.ipynb # Jupyter notebook for testing
├── prompt_library/
│ ├── init.py
│ └── prompt.py # Prompt templates
├── Toolkit/
│ ├── init.py
│ └── toolkits.py # Custom toolkit functions
├── utils/
│ ├── init.py
│ └── lms.py # LLM-related helpers
├── .env # Environment variables (Google API key, etc.)
├── .gitignore
├── agent.py # Main agent logic
├── app.py # Application setup / LangGraph entry
├── LICENSE
├── main.py # FastAPI entry point
├── readme.md # Project documentation
├── requirements.txt # Dependencies
├── setup.py # Package setup


---

## How It Works

### 1. Multi-Agent Workflow
The project uses a **LangGraph** state graph:
[START] → [Supervisor Node] → routes to:
├─ [Booking Node] → handles appointment availability checks
└─ [Information Node] → asks for missing details

- **Supervisor Node** decides where to send the conversation next.
- **Booking Node** checks appointment availability (requires dentist name or specialization).
- **Information Node** collects missing details if Booking Node cannot proceed.

Example conversation flow:
1. **User**: "Can you check if a dentist is available tomorrow at 10 AM?"
2. **Booking Node**: "I am sorry, I need the name of the dentist to check for availability."
3. **Information Node**: "If you don't have a preference for a specific dentist, I can check the availability for a particular specialization. Which dental specialization are you interested in?"

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/multi-agent-doctor-booking.git
cd multi-agent-doctor-booking

2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables
Create a .env file:



