#!/usr/bin/python3

from abc import ABC, abstractmethod;
import onnxruntime as rt;

class PlayerInterface(ABC):
  @abstractmethod
  def get_position(self):
    raise NotImplementedError;
  @abstractmethod
  def get_result_string(self):
    raise NotImplementedError;
  @abstractmethod
  def initialize_game(self, position = None):
    raise NotImplementedError;
  @abstractmethod
  def suggest_move(self, position):
    raise NotImplementedError;
  @abstractmethod
  def play_move(self, c):
    raise NotImplementedError;
  @abstractmethod
  def should_resign(self):
    raise NotImplementedError;
  @abstractmethod
  def to_sgf(self, use_comments = True):
    raise NotImplementedError;
  @abstractmethod
  def set_result(self, winner, was_resign):
    raise NotImplementedError;

class MCTSPlayerInterface(PlayerInterface):
  @abstractmethod
  def get_root(self):
    raise NotImplementedError;
  @abstractmethod
  def tree_search(self, parallel_readout = None):
    raise NotImplementedError;
  @abstractmethod
  def get_num_readouts(self):
    raise NotImplementedError;
  @abstractmethod
  def set_num_readouts(self, readouts):
    raise NotImplementedError;

class MCTSPlayer(MCTSPlayerInterface):
  def __init__(self, onnx_path, seconds_per_move = 5, num_readouts = 0, resign_threshold = None, two_player_mode = False, timed_match = False):
    self.sess = rt.InferenceSession(onnx_path, providers = [])
