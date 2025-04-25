#!/bin/bash
# Simple startup script for the AI Trading Bot
# It loads environment variables and starts the application

# Check if .env file exists and load it
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# Check if we have the required secrets
if [ -z "$SESSION_SECRET" ]; then
  echo "SESSION_SECRET not found! Running setup script first..."
  python setup.py
fi

# Start the application
echo "Starting AI Trading Bot..."
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app