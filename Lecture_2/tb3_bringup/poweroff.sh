#!/bin/bash

source ./config.sh

echo "Powering off system..."
ssh ${RASPBERRY_USERNAME}@${RASPBERRY_IP} "bash -c 'echo ${RASPBERRY_PASSWORD} | sudo -S poweroff'"
