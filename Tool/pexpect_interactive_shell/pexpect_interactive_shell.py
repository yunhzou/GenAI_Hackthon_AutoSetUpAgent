#!/usr/bin/env python3

import pexpect
import sys

class InteractiveShell:
    def __init__(self):
        # Start a clean bash shell without user config files, disabling echo
        self.shell = pexpect.spawn('/bin/bash', encoding='utf-8', timeout=30, echo=False)

        # Send output to stdout for visual feedback
        self.shell.logfile_read = sys.stdout

        # Set a predictable prompt
        self.shell.sendline('export PS1="PROMPT$ "')
        self.shell.expect(r'PROMPT\$ ')
        self.prompt = r'PROMPT\$ '

    def execute(self, command):
        self.shell.sendline(command)
        self.shell.expect(self.prompt)
        # Grab everything before the prompt
        output = self.shell.before.strip()
        return output

    def close(self):
        self.shell.sendline('exit')
        self.shell.close()

if __name__ == "__main__":
    shell = InteractiveShell()

    output_ls = shell.execute('ls')
    print(f"\nLS OUTPUT:\n{output_ls}")

    shell.execute('cd temp')
    output_pwd = shell.execute('pwd')
    print(f"\nPWD AFTER CD:\n{output_pwd}")

    output_echo = shell.execute('ls')
    print(f"\nECHO OUTPUT:\n{output_echo}")

    shell.close()
    print("\nSession closed.")
