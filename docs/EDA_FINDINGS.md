# EDA Findings for Model Preprocessing

This note summarizes the main findings from the Book-Crossing EDA figures in `reports/figures/` and the exported cleaned artifacts in `data/processed/`.

## Dataset Quality

- The catalog contains 271,360 unique book ISBN records.
- Publication years are mostly usable, but 4,634 records have invalid or missing cleaned publication years.
- Book metadata is nearly complete for author and publisher, with only 2 missing values in each field after cleaning.
- The user table contains 278,858 users, but age is incomplete: 110,762 raw ages are missing and 1,248 additional ages are unrealistic.
- After age cleaning, 166,848 users have valid ages. The median valid user age is 32 and the mean is 34.75.

## Catalog Patterns

- The book catalog has a long-tail author and publisher distribution.
- The most represented authors by catalog count are Agatha Christie, William Shakespeare, Stephen King, Ann M. Martin, and Carolyn Keene.
- The most represented publishers are Harlequin, Silhouette, Pocket, Ballantine Books, and Bantam Books.
- Publication years concentrate around modern titles, with a median cleaned publication year of 1996.

## User and Location Patterns

- Country extraction from `Location` creates a useful demographic feature for analysis and optional future modeling.
- The largest user groups are from the USA, Canada, the United Kingdom, Germany, Spain, Australia, and Italy.
- The country distribution is highly concentrated, which may introduce popularity and regional bias in recommendations.

## Rating Behavior

- The ratings table contains 1,149,780 interactions.
- Zero ratings dominate the dataset: 716,109 interactions are implicit feedback and 433,671 are explicit ratings from 1 to 10.
- Explicit ratings are skewed toward positive values, especially ratings 7, 8, 9, and 10.
- Popularity is highly skewed. A small number of users and books contribute a large share of interactions.
- The most active user is `11676` with 13,602 ratings.
- The most rated book is `Wild Animus` with 2,502 ratings.

## Combined Dataset Insights

- The merged table contains 1,149,780 rows, 105,283 users with ratings, 340,556 rated ISBN values, and 327 countries in the exported artifact.
- Popularity-based rankings and average-rating rankings differ substantially, so top-rated lists should use a minimum rating-count threshold.
- In the EDA, books such as `Chobits (Chobits)`, `Free`, and `El Hobbit` rank highly by average rating when requiring at least 20 ratings.
- Country and age-group views show that engagement is not evenly distributed across demographic segments.
- Correlations between rating, age, and publication year are very weak, so these fields should be treated as optional side features rather than strong direct predictors.

## Graph Findings

- The exported user-book graph contains 105,283 user nodes and 340,556 book nodes.
- The interaction graph has 1,149,780 user-book edges before bidirectional expansion.
- The average undirected degree is approximately 5.16.
- The graph density is approximately 0.00003207, confirming that the recommendation problem is very sparse.
- High-degree users and books form hubs that will strongly influence collaborative filtering and LightGCN message passing.

## Preprocessing Implications

- Preserve implicit zero ratings separately or document how they are used, because they represent interaction signals rather than negative preferences.
- Use contiguous user and book node mappings for LightGCN.
- Use bidirectional edges for message passing, with book node indices offset after user node indices.
- Consider filtering sparse users/books for faster experiments, but keep the full exported graph for reproducibility.
- Save cleaned tables and graph artifacts in `data/processed/` so model training can start without rerunning EDA.
