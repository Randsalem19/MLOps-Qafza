# Task 2 EDA Findings - Training Split Only

- Training rows: 67,529; columns: 40.
- Late-delivery prevalence: 9.03%, confirming a materially imbalanced target.
- The strongest univariate numerical association with the label was `customer_seller_distance_km`; this is descriptive, not causal.
- Among states with at least 200 training orders, `AL` had the highest observed late rate.
- Price, freight, item-count, and distance features are skewed; robust imputation and scaling are appropriate.
- Rare product categories require an encoder that safely handles infrequent and unseen values.
- Monthly late rates vary, so purchase month and day-of-week are retained as prediction-time features.
- Post-delivery timestamps, reviews, delivery delay, and order status are excluded from modeling to prevent leakage.
