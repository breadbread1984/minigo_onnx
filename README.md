# MiniGO ONNX version

## Introduction

this project is a port of MiniGO to onnxruntime backend. this port runs on onnxruntime which has a broader compatibility to hardwares.

## Usage

### install prerequisite

```shell
pip3 install -r requirements.txt
sudo apt install default-jre gnugo
```

download and install [gogui](https://sourceforge.net/projects/gogui/).

### how to run

watch MiniGO playing against GnuGO

```shell
gogui -size 19 -program "gogui-twogtp -black \"gnugo --mode gtp\" -white \"python3 play.py\" -games 10 -size 19 -alternate -sgffile gnugo" -computer-both -auto
```

duel with MiniGO

```shell
gogui-twogtp -black 'python3 play.py' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

