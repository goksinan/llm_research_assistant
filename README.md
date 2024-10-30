# Research Paper Assistant CLI

A command-line tool for analyzing academic papers through natural language conversations. Powered by LLMs, it enables researchers to quickly explore and understand academic content through intuitive dialogue.

## Features

- **Document Processing**: Bulk PDF ingestion with smart text extraction and chunking
- **Contextual Chat**: Natural conversations about paper content with source attribution
- **Paper Recommendations**: Suggests related papers from arXiv and Semantic Scholar
- **Multiple LLM Support**: Works with OpenAI GPT-4 and Anthropic Claude
- **Rich Terminal UI**: Clean interface with Markdown support and progress tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant
```

2. Set up Python environment (requires Python 3.9+):
```bash
# Using poetry (recommended)
poetry install

# Using pip
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Required: At least one of these
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
```

## Usage

### Ingest Documents
```bash
# Process a single PDF
cli.py ingest path/to/paper.pdf

# Process multiple PDFs
cli.py ingest paper1.pdf paper2.pdf

# Batch process a directory
cli.py ingest papers/*.pdf --batch
```

### Start Chat Session
```bash
# Open chat with access to all documents
cli.py chat

# Focus on specific documents
cli.py chat -d "Paper Title 1" -d "Paper Title 2"
```

### Chat Commands
- `/help` - Show available commands
- `/documents` - List loaded papers
- `/stats` - Show conversation statistics
- `/focus <title>` - Focus on specific paper
- `/clear` - Clear chat history
- `/quit` - Exit chat


## Project Structure

```
research-assistant/
├── document_processor.py  # PDF processing and chunking
├── conversation.py       # Chat context management
├── llm.py               # LLM provider integration
├── recommendations.py   # Paper recommendation system
├── storage.py           # Vector storage (ChromaDB)
├── templates.py         # Query templates
└── cli.py              # Command-line interface
```

## Dependencies

- `langchain`: Document processing and chunking
- `chromadb`: Vector storage and retrieval
- `openai`: GPT-4 integration
- `anthropic`: Claude integration
- `click`: CLI framework
- `rich`: Terminal formatting
- `pypdf`: PDF extraction
- `arxiv`: Paper recommendations


## License

MIT License - See LICENSE file for details
