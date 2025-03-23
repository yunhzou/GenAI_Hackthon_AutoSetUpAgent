import pexpect
import sys

from Agent import LangGraphAgent
from Tool.tools import *

# Spawn a bash shell
child = pexpect.spawn('/bin/bash', encoding='utf-8')

# Optionally log the session to stdout for debugging
child.logfile = sys.stdout

def process_output(output):
    """
    Agent logic: decide what to do based on output.
    For instance, if the shell waits for a closing quote (dquote> prompt),
    the agent can choose to cancel the command or automatically complete it.
    """
    if 'dquote>' in output:
        print("Agent: Detected an incomplete quote. Sending cancellation signal...")
        child.sendcontrol('c')  # Cancel the current command
    # You can add more logic here based on different outputs or patterns

# Main interaction loop
try:
    while True:
        # Wait for one of the possible outputs: prompt, incomplete quote, or EOF
        index = child.expect([r'\$ ', r'dquote>', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
        
        if index == 0:
            # Standard shell prompt detected
            # The agent can decide to pass control to the user or execute predefined commands
            user_command = input("Enter command: ")
            if user_command.strip().lower() in ['exit', 'quit']:
                child.sendline('exit')
                break
            else:
                child.sendline(user_command)
        elif index == 1:
            # Detected dquote> (incomplete double quote)
            # Let the agent process this output
            process_output(child.before + child.after)
        elif index == 2:
            # EOF detected - shell closed
            print("Agent: Shell session ended.")
            break
        elif index == 3:
            # Timeout reached, you might choose to take an action or simply continue
            print("Agent: Timeout waiting for prompt.")
except KeyboardInterrupt:
    print("\nAgent: Received KeyboardInterrupt, exiting.")
    child.close()