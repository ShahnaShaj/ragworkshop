# Retrieval-Augmented Generation (RAG) with Hugging Face and Gemini

Welcome to the RAG Workshop documentation! This guide will walk you through building a simple Retrieval-Augmented Generation (RAG) system using Hugging Face and Google's Gemini model.

## Overview

RAG enhances the capabilities of Large Language Models (LLMs) by providing them with external information during the text generation process.

### How does it work?

1. **Retrieval**: When you ask a question (a "query"), the system searches a knowledge base (in our case, the text from a PDF you upload) to find the most relevant snippets of text.
2. **Augmentation**: These relevant text snippets are then added to your original query to form a new, more detailed prompt.
3. **Generation**: This augmented prompt is sent to an LLM (like Google's Gemini), which then generates an answer based on the provided context.

This process allows the LLM to answer questions about specific documents it wasn't originally trained on.

## Getting Started

To get started with the workshop, you'll need:

1. A Google Gemini API key
2. Python 3.7 or later
3. Required packages:
      - google-generativeai
      - pypdf
      - sentence-transformers
      - faiss-cpu
      - gradio

## Installation

Install the required packages using pip:

```bash
pip install google-generativeai pypdf sentence-transformers faiss-cpu gradio
```