# LangGraph Research Assistant

A powerful AI research assistant built with LangGraph, FastAPI, and Streamlit. The system implements a modular agent architecture to perform web searches and in-depth research tasks.

## Features

- 🔍 Research agent using TavilySearchResults for detailed web search and analysis
- 🤖 LangGraph-based agent with tool execution and state management
- 🌐 FastAPI backend with:
  - Streaming support using Server-Sent Events
  - Conversation history persistence using SQLite
  - Multi-agent support through `agents.py`
- 💻 Streamlit frontend with:
  - Real-time token streaming
  - Tool execution visualization
  - Chat session sharing and resuming
  - Model selection for OpenAI and Google AI
- 🔄 Support for multiple LLM providers through `settings.py`
- 📊 Visual agent workflow graph generation

## Getting Started

### Prerequisites

- Python 3.11+
- Environment variables:
  - `OPENAI_API_KEY`: OpenAI API key
  - `GOOGLE_API_KEY`: Google API key (optional)
  - Additional provider keys as needed

### Installation

```bash
git clone https://github.com/yourusername/langgraph-research-assistant.git
cd langgraph-research-assistant
pip install -r requirements.txt
```

### Running the Application

1. Start the FastAPI backend:

```bash
python src/run_service.py
```

2. Launch the Streamlit frontend:

```bash
streamlit run src/streamlit_app.py
```

The service components will be available at:

- Backend API: http://localhost:8000
- Streamlit UI: http://localhost:8501

## Demo

<video src="media/Demo_Short.mp4" controls="controls" style="max-width: 100%;">
</video>

## Architecture

The project consists of several key components:

- `research_agent.py`: Core agent implementation with web search capabilities
- `service/`: FastAPI backend with streaming and state management
- `streamlit_app.py`: Interactive frontend with real-time updates
- `core/`: LLM configuration and settings management

## Development

**UPCOMING....**

Use Docker for development with live reload:

```bash
docker compose up --build
```

The service supports development mode through the `MODE=dev` environment variable for automatic reloading of code changes.

## License

MIT License
