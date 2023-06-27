## What is Whisper?

Whisper is an automatic State-of-the-Art speech recognition system from OpenAI that has been trained on 680,000 hours 
of multilingual and multitask supervised data collected from the web. This large and diverse 
dataset leads to improved robustness to accents, background noise and technical language. In 
addition, it enables transcription in multiple languages, as well as translation from those 
languages into English. OpenAI released the models and code to serve as a foundation for building useful
applications that leverage speech recognition. 

## How to start with Docker
1. First of all if you are planning to run the container on your local machine you need to have Docker installed.
You can find the installation instructions [here](https://docs.docker.com/get-docker/).
2. Creating a folder for our files, lets call it `whisper-api`
3. Create a file called requirements.txt and add flask to it.
4. Create a file called Dockerfile 

In the Dockerfile we will add the following lines:

```dockerfile
FROM python:3.10-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install git -y
RUN pip3 install -r requirements.txt
RUN pip3 install "git+https://github.com/openai/whisper.git" 
RUN apt-get install -y ffmpeg

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
```  
### So what is happening exactly in the Dockerfile?
1. Choosing a python 3.10 slim image as our base image.
2. Creating a working directory called `python-docker`
3. Copying our requirements.txt file to the working directory
4. Updating the apt package manager and installing git
5. Installing the requirements from the requirements.txt file
6. installing the whisper package from github.
7. Installing ffmpeg
8. And exposing port 5000 and running the flask server.

## How to create our rout
1. Create a file called app.py where we import all the necessary packages and initialize the flask app and whisper.
2. Add the following lines to the file:

```python
from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = Flask(__name__)
```
3. Now we need to create a route that will accept a post request with a file in it.
4. Add the following lines to the app.py file:

```python
@app.route("/")
def hello():
    return "Whisper Hello World!"


@app.route('/whisper', methods=['POST'])
def handler():
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    # For each file, let's store the results in a list of dictionaries.
    results = []

    # Loop over every file that the user submitted.
    for filename, handle in request.files.items():
        # Create a temporary file.
        # The location of the temporary file is available in `temp.name`.
        temp = NamedTemporaryFile()
        # Write the user's uploaded file to the temporary file.
        # The file will get deleted when it drops out of scope.
        handle.save(temp)
        # Let's get the transcript of the temporary file.
        result = model.transcribe(temp.name)
        # Now we can store the result object for this file.
        results.append({
            'filename': filename,
            'transcript': result['text'],
        })

    # This will be automatically converted to JSON.
    return {'results': results}
```

## How to run the container?
1. Open a terminal and navigate to the folder where you created the files.
2. Run the following command to build the container:

```bash
docker build -t whisper-api .
```
3. Run the following command to run the container:

```bash
docker run -p 5000:5000 whisper-api
```

If you are having errors on MacOS please add `RUN pip3 install markupsafe==2.0.1` to the dockerfile. 

## How to run the container with [Podman](https://podman.io/):

``` bash
cd /tmp
git clone https://github.com/lablab-ai/whisper-api-flask whisper
cd whisper
mv Dockerfile Containerfile
podman build --network="host" -t whisper .
podman run --network="host" -p 5000:5000 whisper
```

Then run:

``` bash
curl -F "file=@/path/to/filename.mp3" http://localhost:5000/whisper
```

Also, from the README:

> In result you should get a JSON object with the transcript in it.

## How to test the API?
1. You can test the API by sending a POST request to the route `http://localhost:5000/whisper` with a file in it. Body should be form-data.
2. You can use the following curl command to test the API:

```bash
curl -F "file=@/path/to/file" http://localhost:5000/whisper
```
3. In result you should get a JSON object with the transcript in it.

## How to deploy the API?
This API can be deployed anywhere where Docker can be used. Just keep in mind that this setup currently using CPU for processing the audio files.
If you want to use GPU you need to change Dockerfile and share the GPU. I won't go into this deeper as this is an introduction.
[Docker GPU](https://docs.docker.com/config/containers/resource_constraints/)

You can find the whole code [here]()

**Thank you** for reading! If you enjoyed this tutorial you can find more and continue reading 
[on our tutorial page](https://lablab.ai/t/)

---

[![Artificial Intelligence Hackathons, tutorials and Boilerplates](https://storage.googleapis.com/lablab-static-eu/images/github/lablab-banner.jpg)](https://lablab.ai)




## Join the LabLab Discord


![Discord Banner 1](https://discordapp.com/api/guilds/877056448956346408/widget.png?style=banner1)  
On lablab discord, we discuss this repo and many other topics related to artificial intelligence! Checkout upcoming [Artificial Intelligence Hackathons](https://lablab.ai) Event


[![Acclerating innovation through acceleration](https://storage.googleapis.com/lablab-static-eu/images/github/nn-group-loggos.jpg)](https://newnative.ai)
