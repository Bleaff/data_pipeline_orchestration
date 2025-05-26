"""
(NEUDC) - A distributed system for video and image processing

NEUDC is a Python-based system for distributed video and image processing. It is designed to
be highly scalable and flexible, allowing you to build complex pipelines for processing
large datasets.

The system is built around a few core components:

- **Nodes**: These are the individual machines that run the system. Each node can run
  multiple **processors**, which are the individual components that perform the actual
  processing.
- **Processors**: These are the individual components that perform the actual processing.
  They can be thought of as functions that take in some input data and produce some output
  data.
- **Queues**: These are the communication channels between the processors. They allow
  processors to send and receive data to and from each other.

The system is designed to be highly extensible, with new processors and nodes able to be
added in as needed.

The system is built on top of several key technologies:

- **ZeroMQ**: This is the messaging system that allows the processors to communicate with
  each other.
- **JAX**: This is the numerical computation library that is used to build the processors.
- **PyTorch**: This is the machine learning library that is used to build some of the
  processors.

The system is designed to be highly scalable, with the ability to run on a single machine
or on a large cluster of machines.

The system is also designed to be highly flexible, with the ability to build complex
pipelines for processing large datasets.

"""
