import os
import subprocess
import pathlib
import shutil

class CommandFailed(Exception):
  def __init__(self, args, result_code, stdout, stderr):
    self.args = args
    self.result_code = result_code
    self.stdout = stdout
    self.stderr = stderr
    super().__init__(self,"\"{}\" failed with exit code:{}".format(" ".join(args), result_code))

def check_result_throw(result):
  if(result.returncode != 0):
    raise CommandFailed(result.args, result.returncode, try_decode(result.stdout), try_decode(result.stderr))

def run_command(command):
  result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  check_result_throw(result)
  return result

SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
SIMULATOR_EXE = os.path.join(SCRIPT_DIR, "build/simulator")
VW_EXE = "vw"
JOINER_EXE = os.path.join(SCRIPT_DIR, "build/reinforcement_learning/test_tools/joiner/joiner.out")

def run_iteration(past_id, current_id):
  if os.path.exists("input.model"):
    pathlib.Path("input.model").unlink()

  # If there is a past model, copy it into the name required for simulator
  if past_id is not None:
    print("\tPast model found, using for simulator")
    dest = shutil.copyfile(past_id + ".model", "input.model")

  print("\tRunning simulator...")
  result = run_command([SIMULATOR_EXE])
  sim_output = result.stdout.decode("utf-8")
  print("\t\tLast line of output: " + sim_output.splitlines()[-1])

  print("\tRunning joiner...")
  result = run_command([JOINER_EXE, "-j", "true"])
  data_file_name = "{}_joined.dsjson".format(current_id)
  with open(data_file_name, 'w') as joined_log:
      joined_log.write(result.stdout.decode("utf-8"))

  print("\tRunning VW...")
  input_model = []
  if past_id is not None:
    input_model = ["-i", past_id + ".model"]
  outut_model = ["-f", current_id + ".model"]
  args = [VW_EXE, "--dsjson", "-d", data_file_name, "--ccb_explore_adf", "--slate", "--epsilon", "0.2", "--power_t", "0", "-l", "0.005", "--save_resume", "--id", current_id]
  args.extend(outut_model)
  args.extend(input_model)
  print("\t\t" + " ".join(args))
  result = run_command(args)

last = None
for i in range(20):
  print("Iteration: " + str(i))
  run_iteration(last, str(i))
  last = str(i)
