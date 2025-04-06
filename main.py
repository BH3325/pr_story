from fastapi import FastAPI, Request, Header, HTTPException
import hmac
import hashlib
import os
import time
import jwt
import base64
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# image upload and comment formatting related imports
from typing import List
import asyncio
import random
from io import BytesIO
from PIL import Image

load_dotenv()

app = FastAPI()


APP_ID = os.getenv("APP_ID")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")  # From GitHub App settings
PK = os.getenv("PK")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

NARRATIVE_PROMPT = "You are given a pull request of git commit messages (professional tone). Convert these messages into a full story. Condense the story into a few broad main actions that can be used as prompts for a text-to-image model. Focus on intuitive metaphors. 1 action per prompt. 77 tokens max. Separately, write the accompanying story (for each prompt) for a human that communicates the metaphor of the image shown and how this contributes to the story whilst not omitting technical details. Here's the diff:\n"

oa_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://hack.funandprofit.ai/api/providers/openai/v1"
)


@app.get("/")
async def hello():
    installations = await get_installations()
    
    for installation in installations:
        print(installation['id'])

    return { "status": "ok" }


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
    return jwt.encode(payload, base64.b64decode(PK), algorithm='RS256')


async def get_installations():
    url = "https://api.github.com/app/installations"
    headers = {
        "Authorization": f"Bearer {generate_jwt()}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

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


async def get_diff(url, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    async with httpx.AsyncClient(follow_redirects=True) as client:
        res = await client.get(
            url,
            headers=headers
        )
        res.raise_for_status()
        return res.text

async def upload_to_imgur(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    headers = {
        "Authorization": f"Client-ID {IMGUR_CLIENT_ID}"
    }

    files = {'image': buffer.getvalue()}

    async with httpx.AsyncClient() as client:
        res = await client.post("https://api.imgur.com/3/image", headers=headers, files=files)
        res.raise_for_status()
        return res.json()["data"]["link"]

async def generate_comment(images: List[Image.Image], captions: List[str]) -> str:
    if len(images) != len(captions):
        raise ValueError("Each image must have a corresponding caption.")

    # Upload images to Imgur
    upload_tasks = [upload_to_imgur(img) for img in images]
    image_urls = await asyncio.gather(*upload_tasks)

    # Build markdown in 2-column layout using raw HTML
    markdown = "<table><tr>"

    for i, (url, caption) in enumerate(zip(image_urls, captions)):
        markdown += f"""
        <td align="center" style="padding: 10px;">
            <img src="{url}" width="300"/><br/>
            <sub>{caption}</sub>
        </td>
        """
        if (i + 1) % 2 == 0:
            markdown += "</tr><tr>"

    markdown += "</tr></table>"
    return markdown

async def handle_pr(payload):
    print(f"processing PR: {payload["pull_request"]["number"]}")

    installation_id = payload["installation"]["id"]
    print(installation_id)

    pr_number = payload["pull_request"]["number"]
    repo_full_name = payload["repository"]["full_name"]

    # get diff
    pr_url = f"https://github.com/{repo_full_name}/pull/{pr_number}.diff"
    token = await get_installation_token(installation_id)
    diff = await get_diff(pr_url, token)

    print(diff)
    # call openai
    completion = oa_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": NARRATIVE_PROMPT + diff
            }
        ]
    )

    print(completion.choices[0].message.content)

    # Endpoint for creating a comment on the PR
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"

    # The comment content
    comment_body = await generate_comment([create_random_image() for _ in range(4)], [f"this is caption {i}" for i in range(4)])

    comment_data = {
        "body": comment_body
    }

    token = await get_installation_token(installation_id)

    # Post the comment to the PR
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # response = requests.post(url, json=comment_data, headers=headers)
    # if response.status_code not in (201, 200):
    #     print(
    #         f"Error posting comment: {response.status_code}, {response.text}")
    #     raise HTTPException(
    #         status_code=500, detail="Failed to post comment to GitHub")

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=comment_data, headers=headers)
        
        if response.status_code not in (200, 201):
            print(f"Error posting comment: {response.status_code}, {response.text}")
            raise HTTPException(status_code=500, detail="Failed to post comment to GitHub")

    return { "status": "ok" }


def create_random_image(size=(512, 512)) -> Image.Image:
    # Generate a random image by setting random colors for each pixel
    image = Image.new("RGB", size)
    pixels = image.load()

    for i in range(image.width):
        for j in range(image.height):
            # Randomize RGB values for each pixel
            pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return image