import subprocess

repetition = 50

for i in range(repetition):
    subprocess.call(['python', 'train.py'])
