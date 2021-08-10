import shlex
import subprocess
cmd = "curl -X 'POST' 'https://coughvid-test-cs2qj3qyha-ew.a.run.app/predict' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'file=@21user2.png;type=image/png'"
args = shlex.split(cmd)
process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

print(stdout)
print(stderr)
print(args)
print(process)