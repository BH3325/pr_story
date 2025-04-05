from fastapi import FastAPI, Request, Header, HTTPException
import hmac
import hashlib
import requests
import os
import time
import jwt
import base64
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

APP_ID = os.getenv("APP_ID")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")  # From GitHub App settings
PRIVATE_KEY = base64.b64decode(os.getenv("PRIVATE_KEY")).decode("utf-8")

@app.get("/")
async def hello():
    sample = generate_jwt()
    print(sample)

    return {"status": "ok"}

@app.post("/webhooks")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None)
):
    if x_hub_signature_256 is None:
        raise HTTPException(status_code=400, detail="Missing signature")

    body = await request.body()

    # Verify the signature
    sha_name, signature = x_hub_signature_256.split("=")
    if sha_name != "sha256":
        raise HTTPException(status_code=400, detail="Unsupported hash type")

    mac = hmac.new(WEBHOOK_SECRET.encode(), msg=body, digestmod=hashlib.sha256)
    if not hmac.compare_digest(mac.hexdigest(), signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = await request.json()

    if x_github_event == "ping":
        return {"msg": "pong"}

    if x_github_event == "pull_request":
        return await handle_pr(payload)

    return {"status": "ok"}

def generate_jwt() -> str:
    now = int(time.time())
    payload = {
        'iat': now - 60,
        'exp': now + (10 * 60),
        'iss': APP_ID,
    }
    return jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')

async def get_installation_token(installation_id: int) -> str:
    jwt_token = generate_jwt()
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers=headers
        )
        res.raise_for_status()
        return res.json()["token"]

async def handle_pr(payload):
    print(f"processing PR: {payload["pull_request"]["number"]}")
    
    pr_number = payload["pull_request"]["number"]
    repo_full_name = payload["repository"]["full_name"]
    
    # Endpoint for creating a comment on the PR
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    
    # The comment content
    comment_data = {
        "body": "yooo"
    }

    installation_id = payload["installation"]["id"]
    print(installation_id)

    token = await get_installation_token(installation_id)

    # Post the comment to the PR
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.post(url, json=comment_data, headers=headers)
    if response.status_code not in (201, 200):
        print(f"Error posting comment: {response.status_code}, {response.text}")
        raise HTTPException(status_code=500, detail="Failed to post comment to GitHub")

    return {"status": "ok"}