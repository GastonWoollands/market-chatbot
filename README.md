# Market Chatbot

## Overview
Market Chatbot is a project that allows you to set up and interact with a market-oriented chatbot. The bot uses natural language processing to answer questions and provide insights related to various market trends.

## Getting Started

### Prerequisites
- Docker installed on your machine
- Access to the Docker image hosted on GitHub

### Setup

To set up the Market Chatbot, you will need to pull the necessary Docker image from the following repository:

```bash
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
