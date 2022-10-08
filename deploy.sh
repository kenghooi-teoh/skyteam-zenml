#!/bin/bash

zenml container-registry register local_registry  --flavor=default --uri=localhost:5000
zenml orchestrator register kubeflow_orchestrator  --flavor=kubeflow
zenml stack register local_kubeflow_stack \
    -m local_metadata_store \
    -a local_artifact_store \
    -o kubeflow_orchestrator \
    -c local_registry

zenml stack set local_kubeflow_stack
zenml stack up