import os 
SYSTEM = os.getenv("SYSTEM")
if SYSTEM == "MAC":
    from .pexpect_interactive_shell_mac import InteractiveShell_Mac as InteractiveShell 
else:
    from .pexpect_interactive_shell import InteractiveShell
