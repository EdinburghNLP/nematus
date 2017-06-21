# Load Testing for Nematus Server

This folder contains a number of specifications for load testing with [Locust](http://www.locust.io).

## Installing Locust

```bash
pip install locustio
```

## Running a test

From within this directory, run

```bash
locust -f test_server_load.py --host http://localhost:8080 --clients=20 --hatch-rate=2 --num-request=100 --no-web
```

This will simulate 20 clients issuing 100 requests altogether, with 2 clients spawned per second. Each request will contain 10 random sentences from German news texts.

Locust offers many more options, including a web-interface for test execution and monitoring. See the [documentation](http://docs.locust.io/en/latest/installation.html) for more information.
