#!/bin/bash

uv run uvicorn app.main:app --reload --host 0.0.0.0
echo "Server started at http://localhost:8000"
