#!/bin/bash

MOUNT_PATH=$EXTERNAL_DRIVE_PATH
OS=$(uname)

if [[ "$OS" == "Linux" ]]; then
  docker run -it --rm --gpus all --ipc=host --net=host --name="ae_perf" \
    -v $(pwd):/app \
    -v $MOUNT_PATH:/app/saves \
    ae_perf
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* || "$OS" == "MSYS"* ]]; then
  WIN_PWD=$(powershell.exe -Command "(Get-Item -Path '.').FullName")
  winpty docker run -it --rm --gpus all --ipc=host --net=host --name="ae_perf" \
    -v $WIN_PWD:/app \
    -v $MOUNT_PATH:/app/saves \
    ae_perf
else
  echo "Unsupported"
fi
read -p "Press enter to continue..."



#xhost +local:root

#docker run -it --rm \
#    --gpus all \
#    --ipc=host \
#    --net=host \
#    --name="ae_perf" \

#    -e DISPLAY=$DISPLAY \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -v ~/.Xauthority:/root/.Xauthority \
#    -v $PWD:/app:rw \
#    -v $EXTERNAL_DRIVE_PATH:/app/saves:rw \
#    ae_perf \
#    /bin/bash 