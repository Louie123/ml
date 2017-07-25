# Toolkit for Document Clustering

> A toolkit for document clustering
>

## Introduction

This is a toolkit for document clustering. It supports three kinds of unsupervised learning methods, e.g., kmeans, single pass cluster, hierarchical clustering. And the visualization for each clustering method is implemented as well .

### Usage

- Parameters and Setting:

  ```shell
  Usage: Toolkit for clustering
  Options:  -h, --help, display the usage of commands
            -c, --corpus, the path of corpus for clustering
            -m, --model [kmeans|hierarchical|SPC], unsupervised learning method for document clustering
            -w, --weight [BOOL|TF|TFIDF], the weight of text representation
  ```

- Example

  ```
   python DCluster.py -c data/corpus -m kmeans -w BOOL
  ```

## Contibutor

Author: Leyi Wang

Email: leyiwang.cn@gmail.com

## Change History

Date: Last update 2017-07-25     Description:  Add the feature selection model.

History: 2017-07-25




