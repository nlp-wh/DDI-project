import subprocess

repetition = 10

for i in range(repetition):
    subprocess.call(['python', 'train.py'])
