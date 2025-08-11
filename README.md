# How Predictable is the Swiss Parliament?  
**Predicting Legislative Voting via Two-Stage Random Walks over a Heterogeneous Graph**  

This repository contains the main code, data, and results from the master's thesis:  
*How Predictable is the Swiss Parliament? Predicting Legislative Voting via Two-Stage Random Walks over a Heterogeneous Graph*.

The project is organized into the following structure:  

---

## Folder Structure

### `data/`
Contains all data required to reproduce the results of the thesis (covering legislative periods 49 to the Spring session 2025 of period 52).

- **clean/**  
  Main data tables including:
  - Votes data
  - Councillors data
  - Affairs data

- **mappings/**  
  Various mapping files used for preprocessing and data linking.

- **gower_weights/**  
  Example Gower weights for the 49th period and co-sponsorship target (periodwise).

- **tuning/**  
  Weight matrices without hold-out and indices of training data (used for within-period tuning in period 49).

- **weight_matrices/**  
  Precomputed weight matrices for the 49th period.

---

### `src/`
All central functions needed to run the **Random Walk over a Heterogeneous Graph (RWHG)** approach.

---

### `notebooks/`
Four example Jupyter notebooks demonstrating how to use the functions from `src/` to run RWHG, illustrated using the 49th legislative period within-matrix prediction setup. Also includes an example notebook which shows how to access the webservice of the Swiss Parliament. 

---

### `results/`
Contains the main results from the evaluations presented in the thesis.

---

## Environment
An `environment.yml` file is provided to set up the required dependencies.  
To create the environment:

```bash
conda env create -f environment.yml
conda activate <environment_name>

