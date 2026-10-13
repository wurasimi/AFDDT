"""Agentic Fused Domain Digital Twin (AFDDT).

The package is organised by the eight layers of the agentic digital twin
reference architecture described in Chapter Three of the dissertation:

    layer1_physical      simulated IoT deployment (devices, edge/fog nodes, gateway)
    layer2_network       message bus (in-process, or a real MQTT broker)
    layer3_data          raw data store, knowledge graph, threat-intelligence ingestion
    layer4_models        detection models, embedding model, language models, simulation
    layer5_xai           explanations
    layer6_services      connection, data, AI/ML and cognitive services
    layer7_agents        the twelve agents and the Orchestrator
    layer8_application   web dashboard

twins.py holds the digital twin objects themselves, and engine.py wires the
layers together.
"""

__version__ = "1.0.0"
