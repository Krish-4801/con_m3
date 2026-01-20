
import requests
import argparse
import sys
import time

SERVER_URL = "http://localhost:8000"

def ingest(video_path, video_id):
    print(f"📤 Sending ingest request for {video_id}...")
    try:
        resp = requests.post(f"{SERVER_URL}/ingest", json={
            "video_path": video_path,
            "video_id": video_id
        })
        if resp.status_code == 200:
            print(f"✅ Success: {resp.json()}")
            print("The server is processing in the background. Check server logs for progress.")
        else:
            print(f"❌ Error {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is 'server.py' running?")

def query(query_text, video_id):
    print(f"🤔 Asking: {query_text}")
    start = time.time()
    try:
        resp = requests.post(f"{SERVER_URL}/query", json={
            "query": query_text,
            "video_id": video_id
        })
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n🤖 Answer ({time.time()-start:.2f}s):\n{data['answer']}\n")
        else:
            print(f"❌ Error {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is 'server.py' running?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conclave Client")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Process a video")
    ingest_parser.add_argument("--video", required=True)
    ingest_parser.add_argument("--id", required=True)

    # Query Command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("--q", required=True)
    query_parser.add_argument("--id", required=True)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.video, args.id)
    elif args.command == "query":
        query(args.q, args.id)
    else:
        parser.print_help()