#!/usr/bin/env python3

import pexpect
import sys

class InteractiveShell:
    def __init__(self, root_dir=None):
        self.shell = pexpect.spawn('/bin/bash', encoding='utf-8', timeout=7200, echo=False)
        self.shell.logfile_read = sys.stdout
        self.shell.sendline('export PS1="PROMPT$ "')
        self.shell.expect(r'PROMPT\$ ')
        self.prompt = r'PROMPT\$ '
        # Set root directory if provided
        if root_dir:
            self.execute(f'cd {root_dir}')

    def execute(self, command):
        if "create" in command:
            if "hf" in command:
                return "Note: ✅ Environment 'hf' already created. Activate directly with 'micromamba activate hf'."
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
