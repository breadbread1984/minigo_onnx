#!/usr/bin/python3

import sys;
from os.path import join, basename;
from absl import app, flags;
import gtp_engine;
from gtp_cmd_handlers import BasicCmdHandler, KgsCmdHandler, GoGuiCmdHandler, MiniguiBasicCmdHandler, RegressionsCmdHandler;
from strategies import CGOSPlayer, MCTSPlayer;
from utils import dbg;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('model_path', default = join('models', 'minigo-op13-fp32-N.onnx'), help = 'path to onnx model');
  flags.DEFINE_enum('mode', enum_values = {'default', 'cgos', 'kgs'}, default = 'default', help = 'game mode');
  flags.DEFINE_enum('device', enum_values = {'cpu', 'gpu'}, default = 'cpu', help = 'device to use');

def main(unused_argv):
  # 1) create player
  player = CGOSPlayer(onnx_path = FLAGS.model_path, seconds_per_move = 5, timed_match = True, two_player_mode = True, device = FLAGS.device) if FLAGS.mode == 'cgos' else \
           MCTSPlayer(onnx_path = FLAGS.model_path, two_player_mode = True);
  # 2) create engine
  engine = gtp_engine.Engine();
  name = 'Minigo-' + basename(FLAGS.model_path);
  version = '0.2';
  engine.add_cmd_handler(gtp_engine.EngineCmdHandler(engine, name, version));
  if FLAGS.mode == 'kgs':
    engine.add_cmd_handler(KgsCmdHandler(player));
  engine.add_cmd_handler(RegressionsCmdHandler(player));
  engine.add_cmd_handler(GoGuiCmdHandler(player));
  engine.add_cmd_handler(BasicCmdHandler(player, courtesy_pass = FLAGS.mode == 'kgs'));
  # 3) play
  dbg("GTP engine ready\n");
  for msg in sys.stdin:
      if not engine.handle_msg(msg.strip()): break;

if __name__ == "__main__":
  add_options();
  app.run(main);
