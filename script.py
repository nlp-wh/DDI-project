import subprocess

repetition = 5

for i in range(repetition):
    subprocess.call(['python', 'train.py'])
