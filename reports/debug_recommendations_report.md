# GraphRec - Recommendation Quality Debug Report

## 1. Book Catalog Genre Distribution

| Genre | Books | % |
|:------|------:|--:|
| Children's Books | 362 | 32.7% |
| Arts, Film & Photography | 127 | 11.5% |
| Business & Economics | 103 | 9.3% |
| Crafts, Home & Lifestyle | 94 | 8.5% |
| Higher Education Textbooks | 89 | 8.0% |
| Biographies, Diaries & True Accounts | 59 | 5.3% |
| Exam Preparation | 55 | 5.0% |
| Fantasy, Horror & Science Fiction | 49 | 4.4% |
| Engineering | 31 | 2.8% |
| Health, Family & Personal Development | 30 | 2.7% |
| Crime, Thriller & Mystery | 29 | 2.6% |
| History | 27 | 2.4% |
| Language, Linguistics & Writing | 23 | 2.1% |
| Law | 22 | 2.0% |
| Literature & Fiction ⚠️ THIN | 7 | 0.6% |

**Total books**: 1,107  
**Thin genres (<20 books)**: 1  
**Dominant genre**: 'Children's Books' at 32.7% of catalog  

## 2. Root Cause Analysis

### Issue 1: Catalog Imbalance
- **'Children's Books'** makes up **32.7%** of the entire catalog (362 books).
- When popularity-weighted sampling is used for exploration, Children's Books dominate cross-persona interactions.
- Persona keywords like 'Romance', 'Sci-Fi' barely match any books by genre name since catalog uses broader Amazon genres.

### Issue 2: Persona Keyword Mismatch
The dataset uses long compound genre names like:
- `Fantasy, Horror & Science Fiction` (not just `Fantasy`)
- `Crime, Thriller & Mystery` (not just `Mystery`)
- `Biographies, Diaries & True Accounts` (not just `Biography`)
This means persona keywords only partially match genre names, reducing the primary pool to near-zero for many personas.

### Issue 3: User 10300
- User 10300 in graph: **False**
- User 10300 ratings found: **0**
- **Root Cause**: The graph was built before this user ID was included in the ratings. The user_mapping.csv only contains 261 users and 10300 may fall outside the range selected.

## 3. Recommended Fixes

### Fix 1: Update Persona Keyword Matching
Update persona primary/secondary genre lists to use the **exact Amazon genre names** in your catalog:
```python
# Use EXACT genre names from Books_Final_Clean.csv:
# "Children's Books"
# "Arts, Film & Photography"
# "Business & Economics"
# "Crafts, Home & Lifestyle"
# "Higher Education Textbooks"
# "Biographies, Diaries & True Accounts"
# "Exam Preparation"
# "Fantasy, Horror & Science Fiction"
# "Engineering"
# "Health, Family & Personal Development"
# "Crime, Thriller & Mystery"
# "History"
# "Language, Linguistics & Writing"
# "Law"
# "Literature & Fiction"
```

### Fix 2: Rebalance Sampling Strategy
- Instead of pure popularity-weighted sampling, use **genre-weighted stratified sampling**.
- Explicitly clamp the % of Children's Books per non-children persona to < 5%.

### Fix 3: Verify User Range in Graph
- User ID range in graph: 10000 to 10260
- Ensure all synthetic users are in Ratings_Final.csv before rebuilding the graph.

