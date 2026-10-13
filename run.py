"""Start the AFDDT and its dashboard.

    python run.py                 start, then open http://127.0.0.1:8000
    python run.py --lan           also reachable from a phone on the same Wi-Fi
    python run.py --port 8080     use another port
"""
import argparse
import os
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Run the Agentic Fused Domain Digital Twin")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--lan", action="store_true", help="listen on all network interfaces")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from afddt.config import SETTINGS
    import uvicorn
    from afddt.layer8_application.server import create_app

    port = args.port or SETTINGS.port
    host = "0.0.0.0" if args.lan else SETTINGS.host
    app = create_app()
    url = f"http://127.0.0.1:{port}"
    print(f"\n[AFDDT] Layer 8  dashboard ready at {url}  (press Ctrl+C to stop)\n")
    if args.lan:
        print("[AFDDT] On your phone, open http://<this computer's IP address>:%d" % port)
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
