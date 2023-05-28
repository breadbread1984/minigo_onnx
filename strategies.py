#!/usr/bin/python3

from abc import ABC, abstractmethod;
import time;
import onnxruntime as rt;
import go;
from utils import dbg;
import symmetries;
from features import *;
import mcts;

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
  def __init__(self, onnx_path, seconds_per_move = 5, resign_threshold = -0.9, two_player_mode = False, timed_match = False, input_feature = 'agz', device = 'cpu'):
    assert -1 <= resign_threshold < 0;
    assert input_feature in {'agz', 'mlperf07'};
    assert device in {'cpu', 'gpu'};
    provider = {'cpu': 'CPUExecutionProvider','gpu': 'CUDAExecutionProvider'}[device];
    self.onnx_path = onnx_path;
    self.network = rt.InferenceSession(onnx_path, providers = [provider]);
    self.input_feature = input_feature;
    self.seconds_per_move = seconds_per_move;
    self.num_readouts = 800 if go.N == 19 else 200;
    self.verbosity = 1;
    self.two_player_mode = two_player_mode;
    if two_player_mode:
      self.temp_threshold = -1;
    else:
      self.temp_threshold = (go.N * go.N // 12) // 2 * 2;
    self.initialize_game();
    self.root = None;
    self.resign_threshold = resign_threshold;
    self.timed_match = timed_match;
    assert (self.timed_match and self.seconds_per_move > 0) or self.num_readouts > 0;
    super(MCTSPlayer, self).__init__();
  def get_position(self):
    return self.root.position if self.root else None;
  def get_root(self):
    return self.root;
  def get_result_string(self):
    return self.result_string;
  def initialize_game(self, position = None):
    if position is None:
      position = go.Position();
    self.root = mcts.MCTSNode(position);
    self.result = 0;
    self.result_string = None;
    self.comments = [];
    self.searches_pi = [];
  def suggest_move(self, position):
    start = time.time();
    if self.timed_match:
      while time.time() - start < self.seconds_per_move:
        self.tree_search();
    else:
      current_readouts = self.root.N;
      while self.root.N < current_readouts + self.num_readouts:
        self.tree_search();
      if self.verbosity > 0:
        dbg("%d: Searched %d times in %.2f seconds\n\n" % (position.n, self.num_readouts, time.time() - start));
    if self.verbosity > 2:
      dbg(self.root.describe());
      dbg('\n\n');
    if self.verbosity > 3:
      dbg(self.root.position);
    return self.pick_move();
  def play_move(self, c):
    if not self.two_player_mode:
      self.searches_pi.append(self.root.children_as_pi(self.root.position.n < self.temp_threshold));
    self.comments.append(self.root.describe());
    try:
      self.root = self.root.maybe_add_child(coords.to_flat(c));
    except go.IllegalMove:
      dbg("Illegal move")
      if not self.two_player_mode:
        self.searches_pi.pop();
      self.comments.pop();
      raise;
    self.position = self.root.position;  # for showboard
    del self.root.parent.children;
    return True;  # GTP requires positive result.
  def pick_move(self):
    if self.root.position.n >= self.temp_threshold:
      fcoord = self.root.best_child();
    else:
      cdf = self.root.children_as_pi(squash=True).cumsum();
      cdf /= cdf[-2];  # Prevents passing via softpick.
      selection = random.random();
      fcoord = cdf.searchsorted(selection);
      assert self.root.child_N[fcoord] != 0;
    return coords.from_flat(fcoord);
  def tree_search(self, parallel_readouts=None):
    if parallel_readouts is None:
      parallel_readouts = min(8, self.num_readouts);
    leaves = [];
    failsafe = 0;
    while len(leaves) < parallel_readouts and failsafe < parallel_readouts * 2:
      failsafe += 1;
      leaf = self.root.select_leaf();
      if self.verbosity >= 4:
        dbg(self.show_path_to_root(leaf));
      # if game is over, override the value estimate with the true score
      if leaf.is_done():
        value = 1 if leaf.position.score() > 0 else -1;
        leaf.backup_value(value, up_to=self.root);
        continue;
      leaf.add_virtual_loss(up_to=self.root);
      leaves.append(leaf);
    if leaves:
      # preprocess
      if self.input_feature == 'agz':
        features = [stone_features, color_to_play_feature],
      elif self.input_feature == 'mlperf07':
        features = [stone_features_4, color_to_play_feature, few_liberties_feature, would_capture_feature]
      positions = [leaf.position for leaf in leaves]
      processed = [np.concatenate([feature(p) for feature in features], axis = 2) for p in positions]
      syms_used, processed = symmetries.randomize_symmetries_feat(processed)
      # predict
      move_probs, values = self.network.run(['policy_output:0', 'value_output:0'], {'pos_tensor:0': processed});
      # postprocess
      move_probs = symmetries.invert_symmetries_pi(syms_used, move_probs)
      
      for leaf, move_prob, value in zip(leaves, move_probs, values):
        leaf.revert_virtual_loss(up_to=self.root);
        leaf.incorporate_results(move_prob, value, up_to=self.root);
    return leaves;
  def show_path_to_root(self, node):
    pos = node.position;
    diff = node.position.n - self.root.position.n;
    if len(pos.recent) == 0:
      return;
    def fmt(move):
      return "{}-{}".format('b' if move.color == go.BLACK else 'w', coords.to_gtp(move.move));
    path = " ".join(fmt(move) for move in pos.recent[-diff:]);
    if node.position.n >= int(go.N ** 2 * 2):
      path += " (depth cutoff reached) %0.1f" % node.position.score();
    elif node.position.is_game_over():
      path += " (game over) %0.1f" % node.position.score();
    return path;
  def is_done(self):
    return self.result != 0 or self.root.is_done();
  def should_resign(self):
    return self.root.Q_perspective < self.resign_threshold;
  def set_result(self, winner, was_resign):
    self.result = winner;
    if was_resign:
      string = 'B+R' if winner == go.BLOCK else "W+R";
    else:
      string = self.root.position.result_string();
    self.result_string = string;
  def to_sgf(self, use_comments = True):
    assert self.result_string is not None;
    pos = self.root.position;
    if use_comments:
      comments = self.comments or ['No comments.'];
      comments[0] = ("Resign Threshold: %0.3f\n" % self.resign_threshold) + comments[0];
    else:
      comments = [];
    return sgf_wrapper.make_sgf(pos.recent, self.result_string,
                                white_name = basename(self.onnx_path) or "Unknown",
                                black_name = basename(self.onnx_path) or "Unknown",
                                comments = comments)
  def extract_data(self):
    assert len(self.searches_pi) == self.root.position.n
    assert self.result != 0
    for pcw, pi in zip(go.replay_position(self.root.position, self.result), self.searches_pi):
      # s_t, a_t, r_t
      yield pwc.position, pi, pwc.result
  def get_num_readouts(self):
    return self.num_readouts
  def set_num_readouts(self, readouts):
    self.num_readouts = readouts

class CGOSPlayer(MCTSPlayer):
  def suggest_move(self, position):
    self.seconds_per_move = time_recommendation(position.n)
    return super().suggest_move(position)
