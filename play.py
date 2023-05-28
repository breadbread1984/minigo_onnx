#!/usr/bin/python3

from absl import app, flags;
from strategies import CGOSPlayer, MCTSPlayer;

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('mode', enum_values = {'cgos', 'kgs'}, default = 'cgos', help = 'game mode');
  flags.DEFINE_enum('device', enum_values = {'cpu', 'gpu'}, default = 'cpu', help = 'device to use');

def main(unused_argv):
  player = CGOSPlayer(network)

if __name__ == "__main__":
  add_options();
  app.run(main);
