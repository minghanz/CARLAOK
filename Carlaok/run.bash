#!/usr/bin/env bash

/home/carla/CARLA_0.9.4/CarlaUE4.sh -benchmark -fps=20 &
pid[0]=$!

sleep 5
python /home/carla/CARLAOK/Carlaok/CARLAOK31.py &
pid[1]=$!

sleep 5
python /home/carla/CARLAOK/Carlaok/cam_client.py &
pid[2]=$!

sleep 5
cd /home/carla/CARLAOK/Carlaok/yolov3
python detect_carla_image_file.py &
pid[3]=$!

trap "kill ${pid[1]}; sleep 5; kill ${pid[1]}; exit 1" INT
wait

#########################33
/home/carla/CARLA_0.9.4/CarlaUE4.sh -benchmark -fps=20

# cd /home/carla/CARLAOK/Carlaok
# python CARLAOK31.py
cd /home/carla/CARLAOK/OpenAI_baselines
python -m baselines.run --alg=deepq --env=carla-v0 --num_timesteps=1e6 --save_path=/home/carla/CARLAOK/Carla_model/generate --load_path=/home/carla/CARLAOK/Carla_model/base_model

cd /home/carla/CARLAOK/Carlaok
python cam_client.py

cd /home/carla/CARLAOK/Carlaok
python lidar_client.py

cd /home/carla/CARLAOK/Carlaok/yolov3
python detect_carla_image_file.py
