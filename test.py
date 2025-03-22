# Dependencies (UV style)
# prompt_toolkit==3.0.39

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import WordCompleter

def main():
    # Define a custom style
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
        'output': 'ansigreen',
        'error': 'ansired',
    })

    # Create key bindings
    bindings = KeyBindings()

    @bindings.add('c-c')
    def _(event):
        """Exit on Ctrl+C."""
        event.app.exit()

    # Word completion example
    commands = ['help', 'exit', 'info', 'run']
    completer = WordCompleter(commands, ignore_case=True)

    # Create a session
    session = PromptSession(completer=completer, key_bindings=bindings, style=style)

    print("Welcome to the Interactive CLI! Type 'help' for commands or 'exit' to quit.")

    while True:
        try:
            # Display the prompt
            user_input = session.prompt("cli> ", style=style, bottom_toolbar="Press Ctrl+C to exit")
            
            # Handle commands
            if user_input == 'exit':
                print("Goodbye!")
                break
            elif user_input == 'help':
                print("Available commands: help, exit, info, run")
            elif user_input == 'info':
                print("This is a simple interactive CLI using prompt_toolkit.")
            elif user_input == 'run':
                print("Running your command...")
            else:
                print(f"Unknown command: {user_input}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()