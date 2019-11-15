import os
import loop

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Simulator MUST output an interaction.fb.data and observation.fb.data
SIMULATOR_EXE = os.path.join(SCRIPT_DIR, "build/cb_skype_slate_simulator")
# Joiner must be able to find the files of the simulator and output a dsjson file
JOINER_EXE = os.path.join(
    SCRIPT_DIR, "build/reinforcement_learning/test_tools/joiner/joiner.out")
# The given arguments must accept the format outputted by the joiner
VW_EXE = "vw"
EXTRA_VW_ARGS = "--cb_explore_adf --epsilon 0.2 --power_t 0 -l 0.005 --save_resume -q UA"


if __name__ == "__main__":
    os.mkdir("cb_sim")
    os.chdir("cb_sim")
    last = None
    for i in range(10):
        id = str(i)
        print("Iteration: {}".format(i))
        loop.run_iteration(last, id, os.path.join("..", SIMULATOR_EXE), os.path.join(
            "..", JOINER_EXE), VW_EXE, EXTRA_VW_ARGS)
        last = id
    os.chdir("..")
