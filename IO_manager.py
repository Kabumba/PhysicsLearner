import bz2
import pickle


class Directories:
    def __init__(self):
        self.BASE_DIR = "../Data"
        self.REPLAY_DIR = self.BASE_DIR + "/Replays"
        self.CORRUPTED_REPLAY_DIR = self.BASE_DIR + "/CorruptedReplays"
        self.UNFINISHED_INPUT_DIR = self.BASE_DIR + "/UnalteredInputSequences"
        self.FINISHED_INPUT_DIR = self.BASE_DIR + "/InterpolatedInputSequences"
        self.OLD_GAME_STATE_DIR = self.BASE_DIR + "/GameTickPackets"
        self.GAME_STATE_DIR = self.BASE_DIR + "/GameStates"
        self.TEST_REPLAY_NAME = "test2"
        self.TEST_REPLAY = self.TEST_REPLAY_NAME + ".replay"
        self.TRAIN_DATA_DIR = self.BASE_DIR + "/Train"
        self.TEST_DATA_DIR = self.BASE_DIR + "/Test"


def load_game_state_sequence(sequence_file):
    dirs = Directories()
    with bz2.BZ2File(dirs.GAME_STATE_DIR + "/" + sequence_file, 'r') as f:
        game_state_sequence = pickle.load(f)
    return game_state_sequence
