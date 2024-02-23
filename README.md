# yes-sir

What if we give an LLM access to a terminal?

This is a small Python experiment that lets an LLM suggest terminal commands and executes them in a loop until the model returns `done`.

## How it works

- Sends goal + command output history to AWS Bedrock (`anthropic.claude-v2:1`)
- Receives the next shell command
- Executes it with Python `subprocess`
- Repeats until completion

## Requirements

- Python 3.8+
- AWS credentials with Bedrock Runtime access in `us-east-1`
- `boto3`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
python app.py
```

The goal is currently hardcoded near the bottom of `app.py`.

## Run with Docker

```bash
docker build -t yes-sir .
docker run --rm \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_SESSION_TOKEN \
  -e AWS_REGION=us-east-1 \
  yes-sir
```

## Safety note

Commands are executed with `shell=True`. Only run in a controlled environment.
