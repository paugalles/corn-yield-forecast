# Corn yield forecasting

This is an exercice of corn yield forecast

## The code

This is the core notebook `3 - AVG SimpleNN - ALLSTATES.ipynb`.
Then the mlruns contains the logs of the trainings.


## Installation & environemnt

Create a python environment, then install the requirements with:

```bash
pip3 install -r requirements.txt
```

Alternatively you can use docker and docker as follows:

1. Build the docker image

```bash
make build
```

2. Raise a running container

```bash
make container
```

3. As soon as the docker is running in background you can launch the following services whithin it:

    * Launch a jupyter lab server

        ```bash
        make nb
        ```

        Then access it from your browser by using this address `localhost:8088/?token=corn`

    * Stops the jupyter server

        ```bash
        make nbstop
        ```

    * Launch an mlflow server

        ```bash
        make mlf
        ```
        Then access it from your browser by using this address `localhost:5055`

    * To raise an interactive shell from our running container

        ```bash
        make execsh
        ```
        
    * To run tests

        ```bash
        make test
        ```

    * Stop the docker container and everything that is running within it

        ```bash
        make stop
        ```
