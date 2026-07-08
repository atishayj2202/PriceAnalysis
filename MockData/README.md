# Mock Data Directory Traits & Specifications

This directory contains simulated datasets covering at least 3 years (156 weeks) on a weekly basis. The mock data models two distinct product categories with unique demand traits and price sensitivities.

---

## 1. Directory Structure

```
MockData/
├── README.md                           # This file
├── electronics/                        # High-value, high-elasticity, tech lifecycle
│   ├── mobile_phone/
│   │   ├── stable/                     # Stable market condition
│   │   ├── inflation/                  # High inflation scenario (CCI drop)
│   │   ├── promo_heavy/                # Frequent marketing promotions
│   │   └── competitor_war/             # Intense pricing pressure from competitors
│   └── laptop/
│       └── [stable, inflation, promo_heavy, competitor_war folders...]
└── fmcg/                               # Low-value, lower-elasticity, seasonal, long lifecycle
    ├── rice/
    │   └── [stable, inflation, promo_heavy, competitor_war folders...]
    ├── shampoo/
    │   └── [stable, inflation, promo_heavy, competitor_war folders...]
    └── face_wash/
        └── [stable, inflation, promo_heavy, competitor_war folders...]
```

---

## 2. Product Category Traits

### 2.1 Electronics (Mobile Phones & Laptops)
- **Base Elasticity (\( e \))**: Highly elastic (\( e \approx -2.5 \) for Mobile Phones, \( e \approx -1.8 \) for Laptops). Consumers are highly price-sensitive and compare specs and prices across retailers.
- **Seasonality (\( S \))**: Low to Moderate. Demand spikes during Holiday seasons (Q4, Black Friday, Cyber Monday) and Back-to-School (August-September).
- **Competitor Sensitivity (\( C \))**: High. A 10% price drop by competitors instantly shifts sales to them unless matched.
- **Product Lifecycle (\( L \))**: Rapid. Products have short cycles:
  - *Launch* (< 6 months, premium early adopters, \( L=0.7 \)).
  - *Growth* (6-18 months, mass market, \( L=0.85 \)).
  - *Mature* (18-36 months, \( L=1.0 \)).
  - *Decline* (> 36 months, clearance, \( L=1.2 \)).
- **Sentiment Sensitivity (\( X \))**: High. During recessions or inflation, discretionary spending on electronics drops significantly.

### 2.2 FMCG (Fast-Moving Consumer Goods: Rice, Shampoo, Face Wash)
- **Base Elasticity (\( e \))**: Inelastic to moderately elastic:
  - *Rice*: Essential staple, highly inelastic (\( e \approx -0.4 \)). People need rice regardless of price.
  - *Shampoo*: Moderate brand loyalty, moderately elastic (\( e \approx -0.8 \)).
  - *Face Wash*: Personal care/cosmetic, more elastic (\( e \approx -1.2 \)).
- **Seasonality (\( S \))**: High for some:
  - *Face Wash / Shampoo*: Summer/Winter shifts (e.g. oily skin products sell more in summer).
  - *Rice*: Stable throughout the year with minor spikes during holiday festivals.
- **Competitor Sensitivity (\( C \))**: Moderate. Consumers might switch brands if there's a big price difference, but brand loyalty acts as a buffer.
- **Product Lifecycle (\( L \))**: Long. FMCG goods stay in the "Mature" phase (\( L=1.0 \)) for years or decades with negligible decay.
- **Sentiment Sensitivity (\( X \))**: Low. FMCG items are daily necessities; sales remain stable even during economic downturns.

---

## 3. Market Conditions (Scenarios)

To facilitate deep testing, each product folder contains files modeling four different market conditions:

### 3.1 Stable Market (`stable/`)
- Baseline scenario with normal price fluctuations (\(\pm 5\%\)).
- Consumer Confidence Index (CCI) is stable at 100.0.
- Normal inventory levels and standard competitive pricing.

### 3.2 Inflationary Scenario (`inflation/`)
- CCI drops from 100.0 down to 85.0 over 3 years.
- Unit cost increases gradually due to supply chain inflation.
- Electronics see a sharp demand contraction (income effect), while FMCG demand holds steady but price sensitivity shifts slightly.

### 3.3 Promo-Heavy Scenario (`promo_heavy/`)
- High frequency of promotional campaigns (week-on-week discount cycles).
- Marketing spend spikes periodically, with corresponding sales lifts (\( 15\% - 45\% \) depending on the product type).

### 3.4 Competitor Price War (`competitor_war/`)
- Key competitors launch an aggressive discount campaign.
- Competitor average price drops by 15% - 25%.
- Own sales drop unless matched, creating a wide price gap.
