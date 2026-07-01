# Dashboard Specification
## E-Commerce Customer Analytics – Power BI / Tableau / Streamlit

---

## KPIs (Card Visuals)

| KPI | Formula | Target |
|-----|---------|--------|
| Total Revenue | SUM(orders[total_amount]) WHERE status='Completed' | MoM +5% |
| Active Customers | COUNT(customers) WHERE churned=0 | >60% of base |
| Churn Rate | SUM(churned) / COUNT(customers) | <40% |
| Average CLV | AVG(clv) WHERE frequency > 0 | >$500 |
| Avg Order Value | SUM(revenue) / COUNT(orders) | >$100 |
| Monthly Retention Rate | Repeat buyers / Total buyers per month | >45% |
| NPS Proxy | AVG(review_rating) | >4.0 |
| Revenue at Risk | SUM(monetary) WHERE segment='At Risk' | Minimise |

---

## Charts Required

### Page 1 – Executive Overview
1. **Revenue Trend** (Line + Bar combo) — Monthly revenue with MoM growth %
2. **Customer Funnel** (Waterfall) — New → Active → Returning → Churned
3. **Revenue by Channel** (Donut) — Website / Mobile / In-Store / Phone
4. **Top 10 Products** (Horizontal Bar) — By total revenue

### Page 2 – Customer Segmentation
5. **RFM Segment Distribution** (Donut / Treemap)
6. **CLV by Segment** (Bar chart)
7. **RFM Scatter Plot** (Scatter) — Recency vs Monetary, colour by segment
8. **Segment Migration** (Sankey / Matrix) — Month-over-month segment changes

### Page 3 – Churn & Retention
9.  **Monthly Retention Rate** (Area line chart)
10. **Cohort Heatmap** (Matrix visual / conditional formatting)
11. **Churn by Segment** (Stacked bar)
12. **Feature Importance** (Bar chart from model output)

### Page 4 – Product & Revenue
13. **Revenue by Category** (Treemap)
14. **Avg Rating by Category** (Bubble chart)
15. **Revenue vs Discount Correlation** (Scatter)
16. **Stock vs Demand** (Dual-axis)

### Page 5 – CLV Deep Dive
17. **CLV Histogram** (Distribution)
18. **CLV vs Frequency** (Scatter)
19. **CLV Projection Table** (Table with conditional formatting)
20. **Top 50 Customers by CLV** (Table)

---

## Slicers / Filters
- Date range picker (order_date)
- Customer segment dropdown
- Product category dropdown
- Channel selector
- State / Region selector

---

## Data Model (Star Schema)
```
              ┌─────────────┐
              │   dim_date  │
              └──────┬──────┘
                     │
┌──────────┐   ┌─────▼──────┐   ┌──────────────┐
│dim_product│◄──│ fact_orders│──►│ dim_customer  │
└──────────┘   └─────┬──────┘   └──────────────┘
                     │
               ┌─────▼──────┐
               │dim_payments│
               └────────────┘
```

## Files to Import into Power BI
1. `data/customers.csv`
2. `data/orders.csv`
3. `data/products.csv`
4. `data/payments.csv`
5. `data/reviews.csv`
6. Feature table exported from notebook: `dashboard/customer_features.csv`

## DAX Measures (Power BI)

```dax
Total Revenue = SUMX(FILTER(orders, orders[order_status]="Completed"), orders[total_amount])

Churn Rate = DIVIDE(COUNTROWS(FILTER(customers, customers[churned]=1)), COUNTROWS(customers))

Avg CLV = AVERAGEX(FILTER(customers, customers[frequency]>0), customers[clv])

MoM Revenue Growth = 
  VAR CurrentMonth = CALCULATE([Total Revenue], DATESMTD('dim_date'[Date]))
  VAR PrevMonth    = CALCULATE([Total Revenue], DATEADD('dim_date'[Date], -1, MONTH))
  RETURN DIVIDE(CurrentMonth - PrevMonth, PrevMonth)

Retention Rate = 
  DIVIDE(
    COUNTROWS(FILTER(orders, orders[is_repeat_customer]=TRUE())),
    DISTINCTCOUNT(orders[customer_id])
  )
```
