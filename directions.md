Modeling the Impact of Congestion Pricing on Ride-Hailing Demand: Segmentation, Elasticity, and Behavioral Shifts

Introduction and Motivation
The prevalence of urban congestion has resulted in long wait / travel times in large metropolitan regions. This congestion is typically confined within certain “popular” zones, and can vary by time of day. As a response, cities are beginning to implement Taxi / Ride-share ‘congestion pricing policies’, or additional fees on travelers when leaving / arriving from these zones. This poses a potential risk for the rideshare business, as these businesses will want to evaluate the cope of customer impact and adjust their matching algorithms and pricing strategies accordingly. 

Our study uses trip-level data to identify behavioral travel segments, estimate price sensitivity, and model how exposure to congestion fees changes travel behavior. Specifically, we examine impacts on trip demand, timing, location choices, service type selection, and the frequency of different trip types. We focus on publicly available New York City Trip Records.
Business problem:
"Quantify the impact of NYC congestion pricing on taxi demand, rider behavior, and revenue to develop data-driven mitigation strategies."

Sub-problems to address:
- Demand Impact: How does congestion pricing affect trip volume in/out of congestion zones?
- Price Sensitivity: What is the elasticity of demand to fare changes?
- Behavioral Changes: Do travelers adjust timing, routes, company election (Uber vs. Traditional taxi) or pickup/dropoff locations to avoid fees?
- Revenue Optimization: What pricing strategies maintain revenue while managing demand?

Stakeholders:
- TLC (Taxi Commission): Revenue/regulation
- Taxis & Uber: Demand prediction, pricing strategy
- Drivers: Earnings, route optimization
- Passengers: Cost, wait times
- City: Traffic management, policy evaluation

Data Description
To provide greater detail on the data, we use trip-level records from the NYC Taxi and Limousine Commission (TLC), covering Yellow Taxi, Green Taxi, For-Hire Vehicle (FHV), and High-Volume For-Hire Vehicle (HVFHV) datasets in PARQUET format. Together, these datasets capture the full spectrum of New York City’s regulated mobility ecosystem, including traditional street-hail taxis, borough-focused services, and app-based ride-hailing platforms such as Uber and Lyft.

Research Design 
We use 2024 data as the pre-policy baseline and 2025 data as the post-policy period, allowing a structured before-and-after comparison of congestion pricing exposure. By combining multiple TLC datasets, we can analyze whether travelers switch transportation modes, change where they travel, or respond differently to price changes depending on service type. This framework allows us to measure not only overall demand shifts, but also changes in who travels, where they travel, when they travel, and which platform they use.

Methodology 
Our team will begin by performing some explorative data to identify any initial patterns, anomalies and redundancies. This effort will be led by Gustavo, who will apply various techniques such as standardization / normalization, initial clustering analysis (to determine types of travelers), as well as univariate and data integrity testing. These processes will help us understand the data better, and will serve as a baseline for the rest of our analysis, and also helps us identify important customer segments that we may want to examine more closely.

After agreeing upon our approach, Hai Hong will spearhead our efforts at applying appropriate feature engineering techniques such as PCA, SVD, and t-SNE to help reduce the curse of dimensionality impacts that might occur. He will also assist in some of our preliminary supervised learning techniques as we attempt to apply various modelling algorithms to determine price sensitivity. 

Along with standard supervised learning, our team will also focus on Random Forest / XGBoost and Boosting algorithms, which will be led by Kevin. These models will focus on pricing sensitivity for our customers, and allowing us to adjust for future congestion taxes (as mentioned in our references) will help us better predict consumer behaviour with respect to future changes.

Evaluation
Evaluation will be led by Ploy, who will be in charge of evaluating socio-economic impact and behavioral validity of the proposed fee structures. She will lead the iterative refinement process, performing "what-if" sensitivity analyses to determine how different pricing fees influence consumer behavior and traffic flow. Furthermore, she is responsible for ensuring the model’s generalizability and fairness. By synthesizing the outputs of the machine learning models with urban planning theory, she ensures that the final recommendations are not just mathematically sound, but also practically implementable and defensible.

Limitations
While the TLC trip-level data provides detailed information on travel behavior and pricing, there are still some limitations. First, the data does not include rider demographics or income, so we cannot directly see how the policy affects different types of people individually.. Second, outside factors such as weather, public transit disruptions, or major city events may affect travel patterns in ways that are hard to fully control for, even when accounting for time and location. Finally, driver decisions and platform algorithm changes are not fully visible in the data, which makes it harder to interpret supply-side responses.


From Professor
Pivot to a matrix that is not time series
Event-based with labelled features


