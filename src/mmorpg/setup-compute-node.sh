#!/bin/bash
# Startup script for GCP VM instance.
#
# Usage:
# 1. Create instance like so (replace <xxx> as appropriate):
#     gcloud compute instances create my-gcp-compute-node --project=gungho-441809
#       --zone=us-central1-a --machine-type=c3d-highcpu-<nCPU> --address=<STATIC-IP-FOR-NORCE-FIREWALL-PASSAGE>
#       --no-restart-on-failure --maintenance-policy=TERMINATE
#       --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration=14400s
#       --service-account=<SOME-NUMBER>-compute@developer.gserviceaccount.com
#       --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append
#       --create-disk=auto-delete=yes,boot=yes,device-name=instance-20241115-160731,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20241016,mode=rw,size=10,type=pd-balanced
#       --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring
#       --labels=goog-ec-src=vm_add-gcloud --reservation-affinity=any
#       --metadata-from-file startup-script=<path-to/setup-compute-node.sh>
#    Wait a moment for installs to finish (e.g. astral-uv below).
# 2. Do
#     gcloud compute config-ssh
# 3. Launch your experiments through `uplink.py:launch()` .
# 4. NB: Remember to delete your instance after you're done!

# Wait for internet connection
while ! ping -c1 google.com >/dev/null; do sleep 1 ; done

apt-get update && apt-get install -y curl neovim less git htop

# Install uv
snap install --classic astral-uv

# The following puts uv in /usr/.local/bin for some reason
# curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sudo sh
