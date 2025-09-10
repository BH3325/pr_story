# PR-Story

PR-Story is a GitHub App that transforms the commit messages of a pull request into a visual narrative. It uses AI to generate a story and accompanying images, which are then posted as a comment on the pull request. This provides a unique and intuitive way to understand the changes introduced in a PR.

## Features

- **Listens for Pull Request Events**: The app is triggered when a new pull request is opened.
- **AI-Powered Narrative Generation**: It uses an OpenAI model to convert commit messages into a compelling story.
- **Text-to-Image Generation**: A Diffusion Pipeline from Hugging Face's `diffusers` library creates images based on the generated story.
- **Automated PR Commenting**: The generated images and their corresponding story captions are posted as a comment on the pull request.

## How It Works

1.  **Webhook Trigger**: A webhook is configured to listen for `pull_request` events from GitHub.
2.  **Fetch Commit Messages**: When a pull request is opened, the app fetches all the commit messages associated with it.
3.  **Generate Story and Prompts**: The commit messages are sent to an OpenAI model (GPT-4o) to generate a narrative. This narrative is broken down into a series of prompts for the text-to-image model.
4.  **Generate Images**: The prompts are fed into a pre-trained text-to-image model (`playgroundai/playground-v2.5-1024px-aesthetic`) to generate a sequence of images.
5.  **Upload Images**: The generated images are uploaded to Imgur to get public URLs.
6.  **Post Comment**: A comment is created on the pull request, containing the images and their corresponding captions, forming a visual story of the changes.

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/BH3325/pr_story.git
    cd pr_story
    ```

2.  **Create a GitHub App**:
    - Go to your GitHub settings and create a new GitHub App.
    - Set the **Webhook URL** to your application's public URL (e.g., `https://your-domain.com/webhooks`).
    - Create a **Webhook Secret** and save it.
    - Under **Permissions**, grant the following repository permissions:
        - **Pull requests**: Read & write
        - **Contents**: Read-only
    - Install the app on your target repositories.
    - Generate a **private key** for your app and download it.

3.  **Set Up Environment Variables**:
    Create a `.env` file in the root of the project and add the following variables:
    ```
    APP_ID="your_github_app_id"
    WEBHOOK_SECRET="your_github_app_webhook_secret"
    PK="your_base64_encoded_private_key"
    OPENAI_API_KEY="your_openai_api_key"
    IMGUR_CLIENT_ID="your_imgur_client_id"
    ```
    - `PK` should be the base64 encoded version of your GitHub App's private key. You can encode it using:
      ```bash
      cat your-private-key.pem | base64
      ```

4.  **Install Dependencies**:
    This project uses `uv` for package management.
    ```bash
    pip install uv
    uv sync
    ```

5.  **Run the Application**:
    Use `uvicorn` to run the FastAPI application:
    ```bash
    uv run uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Usage

Once the app is running and installed on a repository:

1.  Open a new pull request in that repository.
2.  The app will automatically process the commit messages and post a comment with the visual story.

## Future Direction

- Fine tuning: Improve specificity of generated images to better match code context.
- Theme customisation: Allow users to select different art styles for image generation.
- Audio integration: Generate songs or soundscapes to accompany the visual story.