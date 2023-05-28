#!/usr/bin/python3

from os.path import join
from absl import app, flags;
import gtp_engine;
from gtp_cmd_handlers import BasicCmdHandler, KgsCmdHandler, GoGuiCmdHandler
from strategies import CGOSPlayer, MCTSPlayer;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('model_path', default = join('models', 'minigo-op13-fp32-N.onnx'), help = 'path to onnx model');
  flags.DEFINE_enum('mode', enum_values = {'cgos', 'kgs'}, default = 'cgos', help = 'game mode');
  flags.DEFINE_enum('device', enum_values = {'cpu', 'gpu'}, default = 'cpu', help = 'device to use');

def main(unused_argv):
  player = CGOSPlayer(onnx_path = FLAGS.model_path, seconds_per_move = 5, timed_match = True, two_player_mode = True, device = FLAGS.device) if FLAGS.mode == 'cgos' else \
           MCTSPlayer(onnx_path = FLAGS.model_path, two_player_mode = True);
  

if __name__ == "__main__":
  add_options();
  app.run(main);
