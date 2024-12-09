# Network Motifs Reproduction

This repository reproduces the results from the paper "Network Motifs: Simple Building Blocks of Complex Networks" by R. Milo et al. The analysis focuses on identifying significant network motifs in the regulatory networks of *E. coli* and *S. cerevisiae*.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Introduction

Network motifs are recurring, significant patterns of interconnections in complex networks. This project aims to identify and analyze these motifs in biological regulatory networks, specifically in *E. coli* and *S. cerevisiae*.

## Data

### *E. coli* Regulatory Network

- **File**: `data/E_coli_gene_regulation.txt`
- **Format**: Tab-delimited with three columns:
  - **Regulator**: Gene regulating other genes.
  - **Target**: Gene being regulated.
  - **Regulation Type**: 
    - `+` : Positive regulation
    - `-` : Negative regulation
    - `+-` : Dual regulation
- **Source**: [RegulonDB](https://regulondb.ccg.unam.mx/) (Gama-Castro et al., Nucleic Acids Res 36, D120, 2008).

### *S. cerevisiae* Regulatory Network

- **File**: `data/S_cerevisiae_gene_regulation.txt`
- **Format**: Tab-delimited with two columns:
  - **Regulator**: Gene regulating other genes.
  - **Target**: Gene being regulated.
- **Source**: Various biochemical and genetic experiments (Lee et al., Science 298, 799, 2002; Harbison et al., Nature 431, 99, 2004; Horak et al., Genes Dev 16, 3017, 2002; Svetlov & Cooper, Yeast 11, 1439, 1995).

## Environment Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/network-motifs-reproduction.git
   cd network-motifs-reproduction
