# MaCh3 SBI Tools Tutorial!

## Introduction

This is tutorial for [MaCh3SBITools](https://github.com/henry-wallace-phys/MaCh3SbiTools) and aims to provide a practical guide to using MaCh3SBI tools.
It provides a fake physics engine with known results and will walk you through the entire analysis chain! The notebook file [`tutorial_walkthrough.ipynb`](tutorial_walkthrough.ipynb) will walk you through the process of converting physics code to MaCh3SBITools compatible code!

# Building a Simulator

This section will walk you through converting an existing piece of physics code into a compatible simulator

## The "Physics" Engine

The physics engine is designed to mimic the [MaCh3 Python Bindings](https://github.com/mach3-software/MaCh3). It is configurable via [PhysicsConfig.yaml](physics_configs/PhysicsConfig.yaml).

It consists of two components:

- The [`ParameterHandler`](physics_engine/parameter_handler.py). This allows the user to set parameters to some value as well as containing information about the parameter priors
- The [`SampleHandler`](physics_engine/sample_handler.py). This handles the "simulation" step via the `SampleHandler.reweight()` method. It is designed to mimic the behaviour in a proper Monte Carlo reweighting loop.

## Make a Simulator Class

## Storing the Prior

## Running Simulations

## Verifying What You've Done

### Checking the Model

### Checking your "Data" is within the prior constraint

# Training the SBI

## Your first fit

## Monitoring Progress

## Tuning your model

# Inference

# Testing the Model

### Simulation Based Calibration

### Posterior Predictive

### Comparing with the Simulator

#### Comparing Likelihoods

#### Comparing Posteriors

# Presenting Your Results

## Corner Plots

## Posteriors
