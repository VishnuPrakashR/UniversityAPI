# Set base image (host OS)
FROM python:3.11-alpine
RUN pip install --upgrade pip

# Copy the content of the project directory to the working directory
COPY . /university

# Set the working directory in the container
WORKDIR /university

# Install any dependencies
RUN pip install -r requirements.txt

# Specify the Flask environment port
ENV PORT 5000

# By default, listen on port 5000
EXPOSE 5000

# Set the directive to specify the executable that will run when the container is initiated
ENTRYPOINT [ "python" ]

# Specify the command to run on container start
CMD [ "main.py" ]