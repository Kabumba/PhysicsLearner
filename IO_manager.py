import bz2
import pickle

class Directories:
    def __init__(self):
        self.BASE_DIR = "C:/Users/Frederik/Masterarbeit/Data"
        self.REPLAY_DIR = self.BASE_DIR + "/Replays"
        self.CORRUPTED_REPLAY_DIR = self.BASE_DIR + "/CorruptedReplays"
        self.UNFINISHED_INPUT_DIR = self.BASE_DIR + "/UnalteredInputSequences"
        self.FINISHED_INPUT_DIR = self.BASE_DIR + "/InterpolatedInputSequences"
        self.GAME_STATE_DIR = self.BASE_DIR + "/GameTickPackets"
        self.TEST_REPLAY_NAME = "test2"
        self.TEST_REPLAY = self.TEST_REPLAY_NAME + ".replay"
        self.TEST_INTERPOLATOR = "[ConstantSplit,1]"
        self.TEST_INPUT_SEQUENCE = self.TEST_INTERPOLATOR + self.TEST_REPLAY_NAME + "_0.pbz2"

def load_game_state_sequence(sequence_file):
    dirs = Directories()
    with bz2.BZ2File(dirs.GAME_STATE_DIR + "/" + sequence_file, 'r') as f:
        game_state_sequence = pickle.load(f)
    return game_state_sequence