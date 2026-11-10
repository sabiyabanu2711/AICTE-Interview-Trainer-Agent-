#!/usr/bin/env python3
"""
Web server entry point.
Run:  python server.py
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    print(f"\n[*] Interview Trainer Agent running at http://{host}:{port}")
    print("   Open the URL above in your browser.\n")
    app.run(host=host, port=port, debug=debug)
