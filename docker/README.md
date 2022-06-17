# Instructions for running UDE Experiments

## Environment using UDE
The **env using UDE** setting is where both the simulation environment and training algorithm running on the same process and simulation environment interfaced through UDE. 

### Instructions to train a model

1. Build the docker image
```
docker build -t ude_paper .
```
2. Start notebook and OpenAI gym environment inside the same docker container
```
docker run -it -p 3389:3389 -p 8888:8888 -p 80:3003 ude_paper 
```
3. Run the commands inside the docker container that was started from the above command.
``` 
pip install jupyterlab
apt-get install -y unzip
apt-get install -y zip
apt-get install -y ffmpeg libsm6 libxext6
apt-get install -y git

pip install opencv-python
pip install pandas
pip install numpy
pip install torch
pip install matplotlib

# Change the password from ude-demo to something else for more security
jupyter-lab --allow-root --ip=0.0.0.0 --NotebookApp.password="$(echo ude-demo | python3 -c 'from notebook.auth import passwd;print(passwd(input()))')"
```
4. From your web browser you can open jupyter notebook by using the hostname with port 8888
```
https://YOUR_SIMULATION_HOSTNAME:8888
```
5. `ude-demo` will be your default password if you did not make change in step 3.
6. You should find a ipython notebook inside `/Notebooks/with-ude/with_ude.ipynb`
7. Execute the notebook by changing these variables
    i. ENV_NAME = "Hopper-v2"  # This experiment is run for Hopper-v2, LunarLanderContinuous-v2, Pendulum-v1
    ii. ALGO = "PPO"  # Supported are PPO, DDPG, SAC 

## Remote environment using UDE
The **remote env** in local host setting is where the simulation environment is executed on a process different from the training algorithm using UDE’s environment virtualization. Here both processes reside on the same host machine. 

### Instructions to train a model

Build the docker image
```
docker build -t ude_paper .
```

#### Execute in Terminal-1 (Running simulator in one Docker container but same host)
1. Start OpenAI gym environment inside the docker container
```
docker run -it -p 3389:3389 -p 80:3003 ude_paper
```
2. Start the UDE server by executing the following command inside python3
```
python3
ENV_NAME = "Hopper-v2"
from ude_gym_bridge import GymEnvRemoteRunner
env = GymEnvRemoteRunner(env_name=ENV_NAME)
env.start()
```

#### Execute in Terminal-2 (Running notebook in another Docker container but same host)

1. Start Notebook inside this docker container
```
docker run -it -p 8888:8888 ude_paper
```
2. Run the commands inside the docker container that was started from the above command.
``` 
pip install jupyterlab
apt-get install -y unzip
apt-get install -y zip
apt-get install -y ffmpeg libsm6 libxext6
apt-get install -y git
pip install opencv-python
pip install pandas
pip install numpy
pip install torch
pip install matplotlib

# Change the password from ude-demo to something else for more security
jupyter-lab --allow-root --ip=0.0.0.0 --NotebookApp.password="$(echo ude-demo | python3 -c 'from notebook.auth import passwd;print(passwd(input()))')"
```

#### From your Laptop
1. From your web browser you can open jupyter notebook by using the hostname with port 8888
```
https://YOUR_SIMULATION_HOSTNAME:8888
```
5. `ude-demo` will be your default password if you did not make change to default password.
6. You should find a ipython notebook inside `/Notebooks/pseudo-distribution/pseudo-distribution.ipynb`
7. Execute the notebook by changing these variables
    i. HOSTNAME = ""  # Example: HOSTNAME = "ec2-54-221-17-66.compute-1.amazonaws.com". This is where your simulator is running
    ii. ENV_NAME = "Hopper-v2"  # This experiment is run for Hopper-v2, LunarLanderContinuous-v2, Pendulum-v1
    iii. ALGO = "PPO"  # Supported are PPO, DDPG, SAC

## Environment from remote host
The **env from remote host** setting is where the simulation environment and training algorithm are running on different host machines and communicated over the network using UDE’s environment virtualization.

### Instructions to train a model

#### Execute in Host-1 (Running simulator in one host)
1. Build the docker image
```
docker build -t ude_paper .
```
2. Start OpenAI gym environment inside the docker container
```
docker run -it -p 3389:3389 -p 80:3003 ude_paper
```
3. Start the UDE server by executing the following command inside python3
```
python3
ENV_NAME = "Hopper-v2"
from ude_gym_bridge import GymEnvRemoteRunner
env = GymEnvRemoteRunner(env_name=ENV_NAME)
env.start()
```

#### Execute in Host-2 (Running notebook in another host)
1. Build the docker image
```
docker build -t ude_paper .
```
2. Start Notebook inside this docker container
```
docker run -it -p 8888:8888 ude_paper
```
3. Run the commands inside the docker container that was started from the above command.
``` 
pip install jupyterlab
apt-get install -y unzip
apt-get install -y zip
apt-get install -y ffmpeg libsm6 libxext6
apt-get install -y git
pip install opencv-python
pip install pandas
pip install numpy
pip install torch
pip install matplotlib

# Change the password from ude-demo to something else for more security
jupyter-lab --allow-root --ip=0.0.0.0 --NotebookApp.password="$(echo ude-demo | python3 -c 'from notebook.auth import passwd;print(passwd(input()))')"
```

#### From your Laptop
1. From your web browser you can open jupyter notebook by using the hostname where your notebook is running with port 8888
```
https://YOUR_SIMULATION_HOSTNAME:8888
```
5. `ude-demo` will be your default password if you did not make change to default password.
6. Open a blank jupyter notebook and execute the following command
```
!apt-get update
!apt-get install unzip
!unzip ./Notebooks/distribution.zip
```
7. You should find a ipython notebook inside `/Notebooks/distribution/distribution.ipynb`
8. Execute the notebook by changing these variables
    i. HOSTNAME = ""  # Example: HOSTNAME = "ec2-54-221-17-65.compute-1.amazonaws.com". This is where your simulator is running
    ii. ENV_NAME = "Hopper-v2"  # This experiment is run for Hopper-v2, LunarLanderContinuous-v2, Pendulum-v1
    iii. ALGO = "PPO"  # Supported are PPO, DDPG, SAC

### Resource details for UDE experiments

All the experiments were run on EC2 CPU machines. The machine that was choosen to run was `c5.4xlarge` 16 vCPUs and memory 32 GiB.

#### Commands to setup on EC2
1. Basic modules to install on EC2
```
sudo su
apt-get update
apt-get install -y --no-install-recommends ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal
apt update
apt-get install -y xfce4 --no-install-recommends
apt-get install -y xfce4-goodies --no-install-recommends

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update

apt-cache policy docker-ce
apt-get install -y docker-ce

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
```
2. Pull the docker image from ECR
```
docker login -u AWS -p $(aws ecr get-login-password --region us-east-1) https://xxxxxxxx.dkr.ecr.us-east-1.amazonaws.com
docker pull xxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/ude_paper
```
