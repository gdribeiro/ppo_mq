FROM python:3.6

WORKDIR /ppo_app

COPY requirements.txt requirements.txt
RUN apt -y update
RUN apt -y upgrade
RUN apt install -y build-essential
RUN apt install -y libgirepository1.0-dev
RUN pip3 install -v --upgrade pip
RUN pip3 install -r requirements.txt

# COPY . .


# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]