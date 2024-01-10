import subprocess

# Combine the source and make commands into one
command = 'source /home/matheus/p4setup.bash && make'
command2 = 'h3 iperf -s &'
command3 = 'h1 iperf -c h3 -t 10 && h2 iperf -c h3 -t 10'

try:
    subprocess.check_call(command, shell=True, executable='/bin/bash')
    subprocess.check_call(command2, shell=True, executable='/bin/bash')
    subprocess.check_call(command3, shell=True, executable='/bin/bash')
except subprocess.CalledProcessError as e:
    print(f"Error: Command '{command}' failed with return code {e.returncode}.")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")

