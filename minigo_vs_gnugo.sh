#!/bin/bash

gogui -size 19 -program "gogui-twogtp -black \"gnugo --mode gtp\" -white \"python3 play.py\" -games 10 -size 19 -alternate -sgffile gnugo" -computer-both -auto
