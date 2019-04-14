#!/usr/bin/env bash

./CarlaUE4.sh -benchmark -fps=20 &
pid[0]=$!

sleep 5
python manual_control.py &
pid[1]=$!

trap "kill ${pid[1]}; sleep 5; kill ${pid[1]}; exit 1" INT
wait
