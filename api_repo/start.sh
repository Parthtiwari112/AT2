#!/bin/bash
uvicorn app.main_impl:app --host 0.0.0.0 --port $PORT
