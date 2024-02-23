import boto3
import subprocess
import json

# Initialize AWS Bedrock Runtime client
client = boto3.client('bedrock-runtime', region_name='us-east-1')

def execute_command(command):
    """
    Execute a Bash command and return its output.
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return e.output

def get_next_command(output, history, last_command=None):
    """
    Send the output and command history to AWS Bedrock and get the next command to execute,
    with explicit instructions that the response must be a valid Bash command or 'done' 
    if the task is complete. Ensure the prompt format starts with "Human:" as required by the model.
    
    :param output: The output from the last executed command.
    :param history: The history of interaction (prompts and commands) for context.
    :param last_command: The last command that was executed.
    """
    # Ensure the first line of the prompt adheres to the required format
    formatted_history = f"Human: {history}" if not history.startswith("Human:") else history
    
    # Incorporate explicit instructions for the model within the prompt
    instructions = (
        "Respond with a valid Bash command that progresses towards the goal. "
        "Respond only with the plain bash command without any additional text or backticks indicating the start of the code."
        "Know that I always respond with the output of the command. E.g. when your command is `echo 'Hello world.' > test.txt`, I will respond with `` (empty string) because this bash command doesn't return anything. "
        "Please be sure to not get stuck in a loop and try to make progress with each command. "
        "You can check the current state of the system by executing commands like `ls`, `cat`, `pwd`, etc. "
        "Your commands are executed using subprocess in Python, so things like cd, export, etc. will not persist between commands."
        "Use only safe and common Bash commands suitable for a Linux environment. "
        "The whole chat history will be included in the prompt. Please do not send the same commands over and over again. "
        "Do not send the same prompt again. If you do we will end in a loop."
        "IMPORTANT: After executing and validating the result, if the task is fully completed, respond with 'done'. and not with another command. "
        "IMPORTANT: Please do not start an infinite loop with your commands but respond with just 'done' when the result seems okay. No other response is needed."
        "SUPER IMPORTANT!!!!!!!!!: Are you finished? Respond with 'done' if you are finished."
    )

    # If the last command is the same as the next command, append a message to the instructions
    if last_command is not None and last_command == output:
        instructions += " The last command was the same as the previous one. Please provide a different command."

    # Format the prompt to include the history, current output, and the instructions
    # and ensure it starts with "Human:"
    prompt = (
        f"{formatted_history}\n\n"
        f"Human: {output}\n\n"
        f"{instructions}\n\n"
        f"Assistant:"
    )

    body = {
        "prompt": prompt,
        "max_tokens_to_sample": 200,
        "temperature": 0.2,
        "top_p": 1,
        "stop_sequences": ["\n"]
    }

    response = client.invoke_model(
        modelId="anthropic.claude-v2:1",  # Ensure this is the correct model ID
        body=json.dumps(body)
    )
    
    response_body = json.loads(response["body"].read())
    next_command = response_body.get('completion').strip()

    # Update the interaction history to include this latest exchange
    new_history = f"\n\nHuman: {output}\nAssistant: {next_command}"
    history += new_history if not history.endswith(new_history) else ""

    return next_command, history    


def main(goal_description):
    """
    Main function to process the goal description and execute commands until the goal is achieved.
    """
    current_output = goal_description
    history = f"Initial Goal: {goal_description}"  # Start with a clear goal description
    last_command = None
    while True:
        next_command, history = get_next_command(current_output, history, last_command)
        if next_command.lower() in ["done", "exit", "complete"]:
            print("Goal achieved.")
            break
        print(f"🤖 Executing: {next_command}")
        current_output = execute_command(next_command)
        print(f"🖥️  Command Output: {current_output}")
        last_command = next_command

if __name__ == "__main__":
    goal_description = "create a file called test.txt containing 'Hello world.'"
    # goal_description = "create a folder with a random folder name and in that folder a file called test.txt containing 'Hello world.'"
    main(goal_description)
